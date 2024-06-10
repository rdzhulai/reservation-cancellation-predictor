import numpy as np
from metrics import accuracy, hinge_loss


class SVM:
    def __init__(self, lr=0.0001, C=0, tol=1e-7, max_iter=10000, verbose=False):
        """
        Inicializuje model Support Vector Machine (SVM).

        Parametre:
        - lr: Rýchlosť učenia pre gradientný zostup.
        - C: Regularizačný parameter.
        - tol: Tolerancia na určenie konvergencie.
        - max_iter: Maximálny počet iterácií.
        - verbose: Určuje, či sa má počas trénovania vypisovať pokrok.
        """
        self.lr = lr 
        self.C = C 
        self.tol = tol
        self.max_iter = max_iter 
        self.verbose = verbose
        self.w = None

    def decision_function(self, X):
        """
        Vypočíta hodnotu rozhodovacej funkcie pre vstupné dáta.

        Parametre:
        - X: Vstupné dáta.

        Vráti:
        - Hodnoty rozhodovacej funkcie.
        """
        return np.dot(X, self.w[1:]) - self.w[0]

    def cost(self, X, y):
        """
        Vypočíta hodnotu nákladovej funkcie.

        Parametre:
        - X: Vstupné dáta.
        - y: Skutočné značky.

        Vráti:
        - Hodnotu nákladovej funkcie.
        """
        y_pred = self.decision_function(X)
        return self.C * np.linalg.norm(self.w[1:]) ** 2 + np.mean(hinge_loss(y, y_pred))
    
    def gradient(self, x_i, y_i):
        """
        Vypočíta gradient stratovej funkcie.

        Parametre:
        - x_i: Bod vstupných dát.
        - y_i: Skutočná značka pre daný bod.

        Vráti:
        - Vektor gradientu.
        """
        if y_i * self.decision_function(x_i) >= 1: 
            return np.concatenate(([0], 2 * self.C * self.w[1:]))
        else:
            return np.concatenate(([y_i], (2 * self.C * self.w[1:] - y_i * x_i)))
        
    def forward(self, X):
        """
        Vypočíta predpovedané značky pre vstupné dáta.

        Parametre:
        - X: Vstupné dáta.

        Vráti:
        - Predpovedané značky.
        """
        y_pred = self.decision_function(X)
        return np.sign(y_pred)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Natrénuje model SVM na trénovacích dátach.

        Parametre:
        - X: Trénovacie dáta.
        - y: Skutočné značky pre trénovacie dáta.
        - X_val: Validácia dát.
        - y_val: Skutočné značky pre validáciu dát.

        Vráti:
        - Podporné vektory.
        """
        n_features = X.shape[1]
        self.w = np.zeros(n_features + 1)

        count = 0
        prev_cost = float('inf')  # Inicializácia s hodnotou nekonečna pre porovnanie
        while count < self.max_iter:
            for idx, x_i in enumerate(X):
                dw = self.gradient(x_i, y[idx])
                self.w -= self.lr * dw

            cost = self.cost(X, y)
            if abs(prev_cost - cost) < self.tol:
                # Ak je zmena nákladov pod toleranciou, zastaví trénovanie
                break
                
            if self.verbose and count % 10 == 0:
                y_val_pred = self.forward(X_val)
                acc = accuracy(y_val, y_val_pred)
                print(f'Iterácia {count + 1}: vahy = {self.w}, náklad = {cost:.8f}, Správnosť = {acc:.2%}')
            
            prev_cost = cost
            count += 1

        if count == self.max_iter:
            print(f"Počet iterácií presiahol maximálny počet {self.max_iter}")
        else:
            print(f'Iterácia {count + 1}: vahy = {self.w}, náklad = {cost:.8f}, Správnosť = {acc:.2%}')

        # Identifikuje podporné vektory
        support_vector_indices = np.where(np.abs(self.decision_function(X)) <= 1 + 1e-2)[0]
        return X[support_vector_indices]
    