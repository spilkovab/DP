import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class KalmanFilter2D:
    '''
    A 2D Kalman Filter for tracking position and velocity. dv = 0 (constant velocity model).
    '''
    def __init__(self, dt=1.0, process_variance=1e-5, measurement_variance=1e-2):
        # State vector [x, y, vx, vy] where (x, y) is position and (vx, vy) is velocity
        self.x = np.zeros((4, 1))
        
        # State transition matrix for constant velocity model
        # matice A
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Measurement matrix
        # matice H mereni
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Process noise covariance
        # kovariance procesniho sumu
        self.Q = process_variance * np.eye(4)
        
        # Measurement noise covariance
        # kovariance mereni
        self.R = measurement_variance * np.eye(2)
        
        # Estimate error covariance
        self.P = np.eye(4)

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        # Measurement residual
        y = np.asarray(z).reshape(2, 1) - np.dot(self.H, self.x)
        
        # Residual covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate and error covariance
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = (I - np.dot(K, self.H)).dot(self.P)
        
        return self.x[:2].flatten()
    

class KalmanFilterCA:
    '''
    A 2D Kalman Filter with Constant Acceleration model for tracking position, velocity, and acceleration. da = 0.
    '''

    def __init__(self, dt=1.0, process_variance=1e-5, measurement_variance=1e-2):
        # state vector [x, y, vx, vy, ax, ay] where (x, y) is position, (vx, vy) is velocity and (ax, ay) is acceleration
        self.x = np.zeros((6, 1))

        # State transition matrix for constant acceleration model
        self.F = np.array([[1, 0, dt, 0, 0.5*dt*dt, 0],
                           [0, 1, 0, dt, 0, 0.5*dt*dt],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])

        # Process noise covariance
        self.Q = process_variance * np.eye(6)

        # Measurement noise covariance
        self.R = measurement_variance * np.eye(2)

        # Estimate error covariance
        self.P = np.eye(6)

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten()
    
    def update(self, z):
        # Measurement residual
        y = np.asarray(z).reshape(2, 1) - np.dot(self.H, self.x)

        # Residual covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state estimate and error covariance
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = (I - np.dot(K, self.H)).dot(self.P)

        return self.x[:2].flatten() 
        
class simulation:
    def simAndReturn(seed=1):
            np.random.seed(seed)
            dt = 0.5
            steps = 60
            meas_noise_std = np.sqrt(2.0)

            true = np.zeros((6,1))
            true[2,0], true[3,0] = 1.0, 0.8

            kf_cv = KalmanFilter2D(dt=dt, process_variance=1e-2, measurement_variance=2.0)
            kf_ca = KalmanFilterCA(dt=dt, process_variance=1e-2, measurement_variance=2.0)
            kf_cv.x[2,0], kf_cv.x[3,0] = true[2,0], true[3,0]
            kf_ca.x[2,0], kf_ca.x[3,0] = true[2,0], true[3,0]

            truths, measurements, est_cv, est_ca = [], [], [], []

            for t in range(steps):
                acc = (np.random.randn(2) * 0.2).reshape(2,1)
                if np.random.rand() < 0.1:
                    acc = (np.random.randn(2) * 2.0).reshape(2,1)

                true[4:6,0] = acc.ravel()
                true[2:4,0] += true[4:6,0] * dt
                true[0:2,0] += true[2:4,0] * dt + 0.5 * true[4:6,0] * (dt**2)

                meas = true[0:2,0] + np.random.randn(2) * meas_noise_std

                kf_cv.predict(); upd_cv = kf_cv.update(meas)
                kf_ca.predict(); upd_ca = kf_ca.update(meas)

                truths.append(true[0:2,0].copy())
                measurements.append(meas.copy())
                est_cv.append(upd_cv.copy())
                est_ca.append(upd_ca.copy())

            return (np.array(truths), np.array(measurements),
                    np.array(est_cv), np.array(est_ca))
        
    def animate_estimation(truths, measurements, est_cv, est_ca):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlim(np.min(truths[:,0])-1, np.max(truths[:,0])+1)
        ax.set_ylim(np.min(truths[:,1])-1, np.max(truths[:,1])+1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Kalman Filter Estimation Animation")

        # plot elements
        true_line, = ax.plot([], [], 'k-', lw=2, label="True path")
        meas_scatter = ax.scatter([], [], c='gray', alpha=0.5, label="Measurements")
        cv_line, = ax.plot([], [], 'r-', lw=2, label="Kalman CV")
        ca_line, = ax.plot([], [], 'b-', lw=2, label="Kalman CA")
        ax.legend()

        def init():
            true_line.set_data([], [])
            cv_line.set_data([], [])
            ca_line.set_data([], [])
            meas_scatter.set_offsets(np.empty((0,2)))
            return true_line, meas_scatter, cv_line, ca_line

        def update(frame):
            # True path up to current frame
            true_line.set_data(truths[:frame+1,0], truths[:frame+1,1])
            
            # CV and CA estimates up to current frame
            cv_line.set_data(est_cv[:frame+1,0], est_cv[:frame+1,1])
            ca_line.set_data(est_ca[:frame+1,0], est_ca[:frame+1,1])
            
            # Show all past measurements up to current frame
            meas_scatter.set_offsets(measurements[:frame+1])
            
            return true_line, meas_scatter, cv_line, ca_line

        ani = FuncAnimation(fig, update, frames=len(truths),
                            init_func=init, blit=True, interval=100, repeat=False)
        plt.show()

if __name__ == "__main__":
    truths, measurements, est_cv, est_ca = simulation.simAndReturn(seed=1)
    simulation.animate_estimation(truths, measurements, est_cv, est_ca)
