import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        self.mu = 0.02

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]

        # Define layers
        self.uv_layers = uv_layers

        self.loss_rec = []

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf)
        self.f_pred_u, self.f_pred_v, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12, \
            self.f_pred_p = self.net_f(self.x_c_tf, self.y_c_tf)
        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v))\
                      + tf.reduce_mean(tf.square(self.f_pred_s11))\
                      + tf.reduce_mean(tf.square(self.f_pred_s22))\
                      + tf.reduce_mean(tf.square(self.f_pred_s12))\
                      + tf.reduce_mean(tf.square(self.f_pred_p))
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred)) \
                       + tf.reduce_mean(tf.square(self.v_WALL_pred))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_mean(tf.square(self.v_INLET_pred-self.v_INLET_tf))
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred))

        self.loss = self.loss_f + 2*(self.loss_WALL + self.loss_INLET + self.loss_OUTLET)

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10,
                                                                         'maxfun': 10,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = [tf.Variable(tf.random.normal([layers[l], layers[l + 1]], dtype=tf.float32)) for l in range(len(layers) - 1)]
        biases = [tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32) for l in range(len(layers) - 1)]
        return weights, biases

    def xavier_init(self, size):
        in_dim, out_dim = size
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)


    def save_NN(self, fileDir):

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)

        with open(fileDir, 'rb') as file:
            saved_weights, saved_biases = pickle.load(file)

            # Asegurarse de que el modelo almacenado tiene el mismo número de capas
            assert num_layers == len(saved_weights) + 1

            for num, (saved_weight, saved_bias) in enumerate(zip(saved_weights, saved_biases)):
                weight = tf.Variable(saved_weight, dtype=tf.float32)
                bias = tf.Variable(saved_bias, dtype=tf.float32)
                weights.append(weight)
                biases.append(bias)
                print(f" - Parámetros de la NN para la capa {num + 1} cargados exitosamente.")

        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        psi = psips[:,0:1]
        p = psips[:,1:2]
        s11 = psips[:, 2:3]
        s22 = psips[:, 3:4]
        s12 = psips[:, 4:5]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        return u, v, p, s11, s22, s12

    def net_f(self, x, y):

        rho=self.rho
        mu=self.mu
        u, v, p, s11, s22, s12 = self.net_uv(x, y)

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]

        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]

        # f_u:=Sxx_x+Sxy_y
        f_u = rho*(u*u_x + v*u_y) - s11_1 - s12_2
        f_v = rho*(u*v_x + v*v_y) - s12_1 - s22_2

        # f_mass = u_x+v_y

        f_s11 = -p + 2*mu*u_x - s11
        f_s22 = -p + 2*mu*v_y - s22
        f_s12 = mu*(u_y+v_x) - s12

        f_p = p + (s11+s22)/2

        return f_u, f_v, f_s11, f_s22, f_s12, f_p


    def callback(self, loss):
        self.count = self.count+1
        self.loss_rec.append(loss)
        print(f"🚀 ¡Iteración {self.count} completada! Pérdida actual: {loss:.6f} 🎉")


    def train(self, iter, learning_rate):

        tf_dict = {
        self.x_c_tf: self.x_c,
        self.y_c_tf: self.y_c,
        self.x_WALL_tf: self.x_WALL,
        self.y_WALL_tf: self.y_WALL,
        self.x_INLET_tf: self.x_INLET,
        self.y_INLET_tf: self.y_INLET,
        self.u_INLET_tf: self.u_INLET,
        self.v_INLET_tf: self.v_INLET,
        self.x_OUTLET_tf: self.x_OUTLET,
        self.y_OUTLET_tf: self.y_OUTLET,
        self.learning_rate: learning_rate
        }

        loss_records = {
            "WALL": [],
            "INLET": [],
            "OUTLET": [],
            "f": []
        }

        for iteration in range(iter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print every 10 iterations
            if iteration % 10 == 0:
                current_loss = self.sess.run(self.loss, tf_dict)
                print(f"🚀 ¡Iteración {iteration} completada! Pérdida actual: {current_loss:.6f} 🎉")

            loss_records["WALL"].append(self.sess.run(self.loss_WALL, tf_dict))
            loss_records["INLET"].append(self.sess.run(self.loss_INLET, tf_dict))
            loss_records["OUTLET"].append(self.sess.run(self.loss_OUTLET, tf_dict))
            loss_records["f"].append(self.sess.run(self.loss_f, tf_dict))

        self.loss_rec.extend(self.sess.run(self.loss, tf_dict) for _ in range(iter))

        return loss_records["WALL"], loss_records["INLET"], loss_records["OUTLET"], loss_records["f"], self.loss_rec

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):
        feed_dict = {self.x_tf: x_star, self.y_tf: y_star}
        u_star, v_star, p_star = self.sess.run([self.u_pred, self.v_pred, self.p_pred], feed_dict)
        return u_star, v_star, p_star

    def get_loss(self):
        tf_dict = {
            self.x_c_tf: self.x_c, 
            self.y_c_tf: self.y_c,
            self.x_WALL_tf: self.x_WALL, 
            self.y_WALL_tf: self.y_WALL,
            self.x_INLET_tf: self.x_INLET, 
            self.y_INLET_tf: self.y_INLET, 
            self.u_INLET_tf: self.u_INLET, 
            self.v_INLET_tf: self.v_INLET,
            self.x_OUTLET_tf: self.x_OUTLET, 
            self.y_OUTLET_tf: self.y_OUTLET
        }

        loss_records = {
            "WALL": self.sess.run(self.loss_WALL, tf_dict),
            "INLET": self.sess.run(self.loss_INLET, tf_dict),
            "OUTLET": self.sess.run(self.loss_OUTLET, tf_dict),
            "f": self.sess.run(self.loss_f, tf_dict),
            "total": self.sess.run(self.loss, tf_dict)
        }

        return loss_records["WALL"], loss_records["INLET"], loss_records["OUTLET"], loss_records["f"], loss_records["total"]


def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]

def postProcess(xmin, xmax, ymin, ymax, fF, fM, s=2, alpha=0.5, marker='o'):
    [xf, yf, uif, vf, _] = fF
    [xM, yM, uM, vM, _] = fM

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Plot MIXED result
    plot_scatter(ax[0, 0], xM, yM, uM, r'$u$ (m/s)', xmin, xmax, ymin, ymax, s, alpha, marker)
    plot_scatter(ax[1, 0], xM, yM, vM, r'$v$ (m/s)', xmin, xmax, ymin, ymax, s, alpha, marker)

    # Plot FLUENT result
    plot_scatter(ax[0, 1], xf, yf, uif, r'$u$ (m/s)', xmin, xmax, ymin, ymax, s, alpha, marker)
    plot_scatter(ax[1, 1], xf, yf, vf, r'$v$ (m/s)', xmin, xmax, ymin, ymax, s, alpha, marker)

    plt.savefig('./uv.png', dpi=300)
    plt.close('all')

def plot_scatter(ax, x, y, data, title, xmin, xmax, ymin, ymax, s, alpha, marker):
    cf = ax.scatter(x, y, c=data, alpha=alpha-0.1, edgecolors='none', cmap='viridis', marker=marker, s=int(s))
    ax.axis('square')
    for key, spine in ax.spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title)
    ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

def preprocess(dir='FenicsSol.mat'):
    '''
    Load reference solution from Fenics or Fluent
    '''
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    P = data['p']
    vx = data['vx']
    vy = data['vy']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    p_star = P.flatten()[:, None]
    vx_star = vx.flatten()[:, None]
    vy_star = vy.flatten()[:, None]

    return x_star, y_star, vx_star, vy_star, p_star

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def greatest_common_divisor(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def _custom_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = greatest_common_divisor(num, den)
        (num, den) = (int(num / com), int(den / com))
        
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)
    
    return _custom_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


if __name__ == "__main__":

    # Definición de límites del dominio
    lb = np.array([0, 0])
    ub = np.array([1.1, 0.41])

    # Configuración de la red neuronal
    uv_layers = [2] + 8*[40] + [5]

    # Pared = [x, y], u=v=0
    wall_up = [0.0, 0.41] + [1.1, 0.0] * lhs(2, 441)
    wall_lw = [0.0, 0.00] + [1.1, 0.0] * lhs(2, 441)

    # Entrada = [x, y, u, v]
    U_max = 1.0
    INLET = [0.0, 0.0] + [0.0, 0.41] * lhs(2, 201)
    yInlet = INLET[:, 1:2]
    uInlet = 4 * U_max * yInlet * (0.41 - yInlet) / (0.41**2)
    vInlet = 0 * yInlet
    INLET = np.concatenate((INLET, uInlet, vInlet), 1)

    # Salida = [x, y], p=0
    OUTLET = [1.1, 0.0] + [0.0, 0.41] * lhs(2, 201)

    # Superficie del cilindro
    r = 0.05
    theta = [0.0] + [2 * np.pi] * lhs(1, 251)
    x_CYLD = np.multiply(r, np.cos(theta)) + 0.2
    y_CYLD = np.multiply(r, np.sin(theta)) + 0.2
    CYLD = np.concatenate((x_CYLD, y_CYLD), 1)

    WALL = np.concatenate((CYLD, wall_up, wall_lw), 0)

    # Puntos de colocación para el residual de la ecuación
    XY_c = lb + (ub - lb) * lhs(2, 40000)
    XY_c_refine = [0.1, 0.1] + [0.2, 0.2] * lhs(2, 10000)
    XY_c = np.concatenate((XY_c, XY_c_refine), 0)
    XY_c = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)

    XY_c = np.concatenate((XY_c, WALL, CYLD, OUTLET, INLET[:, 0:2]), 0)

    print(XY_c.shape)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue')
    plt.scatter(WALL[:,0:1], WALL[:,1:2], marker='o', alpha=0.2 , color='green')
    plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='o', alpha=0.2, color='orange')
    plt.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='o', alpha=0.2, color='red')
    plt.show()

    with tf.device('/device:CPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Inicialización y carga del modelo
        # model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub)
        model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel=1, uvDir='Pinn/uvNN.pickle')

        # Entrenamiento del modelo
        start_time = time.time()
        loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(iter=1, learning_rate=5e-4)
        model.train_bfgs()
        elapsed_time = time.time() - start_time
        print(f"Entrenamiento completado en {elapsed_time:.2f} segundos")

        # Guardar el modelo y el historial de pérdidas
        model.save_NN('../uvNN.pickle')
        with open('loss_history.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)

    # Cargar resultados de Fluent
    [xf, yf, uf, vf, pf] = preprocess(dir='ReferenceMat/FluentSol.mat')
    fF = [xf, yf, uf, vf, pf]

    # Obtener predicción de PINN en forma mixta
    xpinn = np.linspace(0, 1.1, 251)
    ypinn = np.linspace(0, 0.41, 101)
    xpinn, ypinn = np.meshgrid(xpinn, ypinn)
    xpinn, ypinn = xpinn.flatten()[:, None], ypinn.flatten()[:, None]
    dst = ((xpinn-0.2)**2 + (ypinn-0.2)**2)**0.5
    xpinn, ypinn = xpinn[dst >= 0.05], ypinn[dst >= 0.05]
    xpinn, ypinn = xpinn.flatten()[:, None], ypinn.flatten()[:, None]
    upinn, vpinn, ppinn = model.predict(xpinn, ypinn)
    fM = [xpinn, ypinn, upinn, vpinn, ppinn]

    # Comparación y visualización de u, v, p
    postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, fF=fF, fM=fM, s=3, alpha=0.5)