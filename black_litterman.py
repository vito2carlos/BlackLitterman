import numpy as np
import pandas as pd
from frontiere_efficiente import FrontiereEfficiente

masi_w = pd.read_csv('data/masi_w').set_index('Unnamed: 0')['0']

class BlackLitterman:
    """
    L'objet BlackLitterman exige des inputs spécifiques et propose des méthodes
    pour calculer les rendements espérés et la matrice de covariance
    en utilisant le modèle Black-Litterman.

    -Inputs:

        "sigma":    la matrice de covariance
                    type: (NxN) pd.DataFrame
        "Q":        le vecteur des anticipations
                    type: (Kx1) np.ndarray
        "P":        la matrice des poids des anticipations (matrice de passage)
                    type: (KxN) np.ndarray
        "omega":    la matrice (diagonale) d'incertitude sur les anticipations
                    type: (KxK) np.ndarray
        "tau":      le facteur d'incertitude sur le marché
                    tau est fort => l'incertitude est forte
                    type: float

        "marche_poids":         les pondérations des actifs dans le portefeuille de marché
                                type: pd.DataFrame ou pd.Series
        "prime_de_risque":      la prime de risque
                                type: float
        "taux_sans_risque":     le rendement de l'actif non risqué
                                type: float
    
    -Methodes:

        "rendements()":             calculer les post-rendements de Black-Litterman
                                    E(R)= [(τΣ)⁻¹ + P'.Ω⁻¹.P ]⁻¹.[(τΣ)⁻¹.π + P'.Ω⁻¹.Q]
                                        =  π + τΣ.P'.[Ω + P.τΣ.P']⁻¹.[Q - P.π]
                                    :return:    les rendements espérés
                                    :rtype:     pd.DataFrame
        "cov()":                    calculer la post-matrice de covariance de Black-Litterman
                                    Σ + [(τΣ)⁻¹ + P'.Ω⁻¹.P ]⁻¹
                                    :return:    la matrice de covariance
                                    :rtype:     pd.DataFrame
        "frontiere_efficiente()":   calculer la frontière efficiente apartir des rendements
                                    et de la matrice de covariance de Black-Litterman
                                    :return:    un objet de la classe FrontiereEfficiente
                                    :rtype:     FrontiereEfficiente()
        "portefeuille()":           calculer les pondérations optimales sans contraintes
                                    E(R) = δΣ.w   =>   w = (δΣ)⁻¹.E(R)
                                    :return:    le portefeuille optimal sans contraintes
                                    :rtype:     pd.DataFrame
    """
    def __init__(
        self,
        sigma,
        Q=None,
        P=None,
        omega=None,
        tau=0.07027888706,
        marche_poids=masi_w,
        prime_de_risque=0.05,
        taux_sans_risque=0.02,
        ):
        self.prime_de_risque = prime_de_risque
        self.taux_sans_risque = taux_sans_risque
        self.sigma = sigma
        self.tau = tau
        self.Q = Q
        self.P = P
        self.omega = omega
        self.marche_poids = marche_poids
        self.delta = self.prime_de_risque / (self.marche_poids.T @ self.sigma @ self.marche_poids)
        pi = (self.delta * self.sigma @ self.marche_poids) + self.taux_sans_risque
        self.pi = pd.DataFrame(pi)
        self.pi.columns= ['Rendements de marché']
        self.rends = None
        self.cov_bl = None
        self.poids = None

    def rendements(self):
        if isinstance(self.P, np.ndarray) and isinstance(self.Q, np.ndarray)\
            and self.P.shape[0]*self.P.shape[1] != 0 and self.Q.shape[0]*self.Q.shape[1] != 0:
            if not isinstance(self.omega, np.ndarray):
                self.omega = self.tau * self.P @ self.sigma @ self.P.T
            pi = self.pi.values
            tau_sigma_P = self.tau * self.sigma @ self.P.T    
            A = self.omega + (self.P @ tau_sigma_P)
            b = self.Q - self.P @ pi
            post_rends = pi + tau_sigma_P @ np.linalg.solve(A, b)
            post_rends = pd.DataFrame(post_rends)
        else:
            post_rends = self.pi.copy()
        post_rends.columns = ['Rendements de Black-Litterman']
        self.rends = post_rends
        return post_rends
    
    def cov(self):        
        if isinstance(self.omega, np.ndarray) and self.omega.shape[0]*self.omega.shape[1] != 0:
            tau_sigma_inv = np.linalg.inv(self.tau * self.sigma)
            p_omega_inv = self.P.T @ np.linalg.inv(self.omega) @ self.P
            dev = np.linalg.inv(tau_sigma_inv + p_omega_inv)
            cov_bl = self.sigma + dev
            cov_bl = pd.DataFrame(cov_bl, index=self.sigma.index, columns=self.sigma.columns)
        else:
            cov_bl = self.sigma
        self.cov_bl = cov_bl
        return cov_bl
    
    def frontiere_efficiente(self):
        if not isinstance(self.cov_bl, pd.DataFrame):
            self.cov_bl = self.cov()
        if not isinstance(self.rends, (pd.DataFrame, pd.Series)):
            self.rends = self.rendements()
        return FrontiereEfficiente(self.rends, self.cov_bl, taux_sans_risque=self.taux_sans_risque, prime_de_risque=self.prime_de_risque, marche_poids=self.marche_poids)
    
    def portefeuille(self):
        if not isinstance(self.cov_bl, pd.DataFrame):
            self.cov()
        if not isinstance(self.rends, pd.DataFrame):
            self.rendements()
        A = self.delta * self.cov_bl
        b = self.rends
        poids_bl = np.linalg.solve(A, b)
        self.poids_bl = pd.DataFrame(poids_bl.round(16), index=self.sigma.index, columns=['Poids de Black-Litterman'])
        return self.poids_bl

