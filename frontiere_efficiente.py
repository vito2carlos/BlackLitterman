import numpy as np
import pandas as pd
import cvxpy as cp
import math

masi_w = pd.read_csv('data/masi_w').set_index('Unnamed: 0')['0']

frequence = 252
def rendements(cours):
    """
    Calculer les rendements apartir des cours.

    :param cours: les cours (journaliers) ajustés des actions
    :type cours: pd.DataFrame ou np.ndarray

    :return: les rendements (journaliers)
    :rtype: pd.DataFrame ou np.ndarray
    """
    return cours.pct_change().dropna(how="all")

def moyenne_historique(cours):
    """
    Calculer la moyenne géométrique des rendements annuels
    apartir des rendements (journaliers).
 
    :param cours: les cours (journaliers) ajustés des actions
    :type cours: pd.DataFrame ou np.ndarray

    :return: les rendements (journaliers)
    :rtype: pd.DataFrame ou np.ndarray
    """
    rends = rendements(cours)
    return (1 + rends).prod() ** (frequence / rends.count()) - 1

def fixer_linversibilite(matrice):
    if np.all(np.linalg.eigvals(matrice) > 0):
        return matrice
    q, P = np.linalg.eigh(matrice)
    q = np.where(q > 0, q, 0)
    rmatrice = P @ np.diag(q) @ P.T
    return pd.DataFrame(rmatrice, columns=list(matrice.index), index=list(matrice.index))

def matrice_covariance(cours, cours_rendements=False):
    """
    Calculer la matrice de covariance des rendements.
 
    :param cours: les prix (journaliers) ajustés des actions
    :type cours: pd.DataFrame ou np.ndarray
    :param rendements: True si le premier parametre est les rendements
    :type rendements: bool

    :return: la matrice de covariance
    :rtype: pd.DataFrame ou np.ndarray
    """
    if cours_rendements:
        rends = cours
    else:
        rends = rendements(cours)
    cov = rends.cov() * frequence
    return fixer_linversibilite(cov)

class FrontiereEfficiente:
    """
    L'objet FrontièreEfficiente propose des méthodes pour calculer
    des portefeuilles sur la frontière efficiente.
    
    -Inputs:

        "rendements":           les rendements annuels esperés des actions
                                type: pd.DataFrame ou np.ndarray
        "cov_matrice":          la matrice de covariance
                                type: pd.DataFrame ou np.ndarray
        "prime_de_risque":      la prime de risque
                                type: float
        "taux_sans_risque":     le rendement de l'actif non risqué
                                type: float
        "n_actions":            le nombre des actions
                                type: int

    -Methodes:

        "variance_minimale()":      minimiser la variance du portefeuille
                                    min w'.Σ.w 
                                    :return:    le portefeuille de variance minimale
                                    :rtype:     pd.Series
        "rendement_maximal()":      maximiser le rendement du portefeuille
                                    max w'.π
                                    :return:    le portefeuille de rendement maximal
                                    :rtype:     pd.Series
        "portefeuille_optimal()":   optimiser la fonction d'utilité quadratique
                                    min w'.π - λw'.Σ.w 
                                    :return:    le portefeuille optimal
                                    :rtype:     pd.Series
        "sans_contraintes()":       calculer les pondérations optimales sans contraintes
                                    E(R) = δΣ.w   =>   w = (δΣ)⁻¹.E(R)
                                    :return:    le portefeuille optimal sans contraintes
                                    :rtype:     pd.DataFrame

        "rendement()":              calculer le rendement du portefeuille
                                    :return:    le rendement du portefeuille
                                    :rtype:     float
        "risque()":                 calculer la variance du portefeuille
                                    :return:    le risque du portefeuille
                                    :rtype:     float
        "ratio_sharpe()":           calculer le ratio de sharpe
                                    :return:    le ratio de sharqe du portefeuille
                                    :rtype:     float
    """
    def __init__(
        self,
        rendements=None,
        cov_matrice=None,
        marche=None,
        marche_poids=masi_w,
        prime_de_risque=0.03,
        taux_sans_risque=0.02,
        betas=None,
        n_actions=None
        ):
        self.rendements = rendements
        if isinstance(cov_matrice, (pd.DataFrame, pd.Series)):
            self.cov_matrice = cov_matrice
        elif isinstance(rendements, (pd.DataFrame, pd.Series)):
            self.cov_matrice = matrice_covariance(self.rendements, rendements=True)
        if isinstance(marche, (pd.DataFrame, pd.Series)):
            marche = rendements(marche)
            self.aversion_au_risque = prime_de_risque / (marche.var() * frequence)
        else:
            self.aversion_au_risque = prime_de_risque / (marche_poids.T @ self.cov_matrice @ marche_poids)
        self.taux_sans_risque = taux_sans_risque
        if not isinstance(rendements, (pd.DataFrame, pd.Series)):
            self.rendements = (self.aversion_au_risque * self.cov_matrice @ marche_poids) + self.taux_sans_risque
        if isinstance(betas, (pd.DataFrame, pd.Series)):
            self.betas = betas
        else:
            betas = (self.cov_matrice @ marche_poids) / (marche_poids.T @ self.cov_matrice @ marche_poids)
            betas = pd.DataFrame(betas, index=cov_matrice.columns)
            betas.columns = ['Betas']
            self.betas = round(betas, 1)
        if n_actions is None:
            self.n = len(rendements)
        else:
            self.n = n_actions
        self.poids = None

    def solver(self, objective):
        w = cp.Variable(self.n)
        probleme = cp.Problem(cp.Minimize(objective(w)),
            [
            #les contraintes: 
            cp.sum(w) == 1,#1: la somme des poids est 1
            w >= np.zeros(self.n),#2: le shorting n'est pas possible
            w <= np.concatenate((np.array([0.20, 0.15]),0.1*np.ones(self.n - 2))),#3: les limites des poids
            ])
        probleme.solve()
        self.poids = pd.Series(w.value.round(16), masi_w.index)
        
    def variance_minimale(self):
        objective = lambda w : cp.quad_form(w, self.cov_matrice)
        self.solver(objective)
        return self.poids

    def rendement_maximal(self):
        objective = lambda w : - w @ self.rendements
        self.solver(objective)
        return self.poids

    def portefeuille_optimal(self):
        risque = lambda w : 0.5 * self.aversion_au_risque * cp.quad_form(w, self.cov_matrice)
        objective = lambda w : -(w @ self.rendements) + risque(w)
        self.solver(objective)
        return self.poids
    
    def sans_contraintes(self):
        A = self.aversion_au_risque * self.cov_matrice
        b = self.rendements
        poids = np.linalg.solve(A, b)
        return pd.Series(poids, masi_w.index)

    def rendement(self):
        if not isinstance(self.poids, pd.Series):
            self.portefeuille_optimal()
        return self.poids.T @ self.rendements
    
    def risque(self):
        if not isinstance(self.poids, pd.Series):
            self.portefeuille_optimal()
        return math.sqrt(self.poids.T @ self.cov_matrice @ self.poids)
    
    def ratio_sharpe(self):
        r = self.rendement()
        s = self.risque()
        return (r - self.taux_sans_risque) / s

















