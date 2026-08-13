
## Avant desole si j'ai pas mis les accents ou les apostrophes je suis sur un clavier qwerty et c'est pas facile de les mettre je le ferai plus tard si j'ai le temps

# je dois faire une regression lineaire simple avec numpy 
#on sait que la regression lineaire simple est de la forme y = ax + b

import numpy as np

class regression_lineaire: # ici je vais juste definir a qui represente la pente et b qui represente l'ordonnée à l'origine ou l'intercept si vous voulez
    def __init__(self):
        self.a=None
        self.b=None
    # on va calculer nous meme a et b 
    def fit(self,X,Y):
        #la formule pour calculer a est a = (x_i - x_bar) * (y_i - y_bar) / (x_i - x_bar)^2 et pour b c'est b = y_bar - a * x_bar
        #alors on va calculer x_bar et y_bar qui sont les moyennes de X et Y 
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        
        numerator = np.sum((X - X_mean) * (Y - Y_mean))     
        denominator = np.sum((X - X_mean) ** 2)
        
        #j'ai calculé le numérateur de la formule de a et le dénominateur de la formule de a chacun de son coté pour me simplifier la tache
        
        self.a = numerator / denominator
        self.b = Y_mean - self.a * X_mean

    def predict(self, X):
        return self.a * X + self.b
    
    #cette approche est unique utilisable lorsque X et Y sont des arrays numpy 1D, si vous voulez utiliser des arrays numpy 2D il faudra faire quelques modifications dans le code pour la regrssion multiple la prochaine fois je le ferai pour le moment je suis pas encore pret pour la regression multiple :(
        

# mais bon et si on testait  
np.random.seed(42)  # pour avoir toujours les mêmes résultats       
X = np.linspace(0, 10, 30)
Y = 3 * X + 5 + np.random.normal(0, 1, size=X.shape) # j'ai choisit 3 pour a et 5 pour b au hazard  

model = regression_lineaire()
model.fit(X, Y)

print(model.a, model.b)

#je vais le comparer avec sklearn pour voir si j'ai bien fait mon taf

#j'avoue aussi que je me suis un peu aide de claude pour quelques parties de code comme ajouter le bruit avec np.random.normal(0, 1, size=X.shape) j'avais oublie comment faire , aussi pour la comparaison avec le vrai modele 

from sklearn.linear_model import LinearRegression

sk_model = LinearRegression()
sk_model.fit(X.reshape(-1, 1), Y)
print(sk_model.coef_[0], sk_model.intercept_)
 
 
 #donc normalement notre modele devrait etre tres proche de celui de sklearn reste plus a tester  avec (python RL.py) et voir si ca marche bien
 # ca marche tres bien je suis content de moi