import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def simulation_chaleur(L, alpha, N, r, t_fin, T_gauche, T_droite, T_init, save_frac=10):
    """
    Résout l'équation de la chaleur 1D par schéma FTCS explicite (version avec boucle).
    
    Paramètres :
    ------------
    L          : longueur de la barre (m)
    alpha      : diffusivité thermique (m²/s)
    N          : nombre d'intervalles spatiaux
    r          : nombre de Fourier discret (r <= 0.5 pour la stabilité)
    t_fin      : durée totale de simulation (s)
    T_gauche   : température à x=0 (°C)
    T_droite   : température à x=L (°C)
    T_init     : température initiale uniforme (°C)
    save_frac  : fraction de sauvegarde
    
    Retourne :
    ----------
    x, t_save, T_save
    """
    dx = L / N
    dt = r * dx**2 / alpha
    nt = int(t_fin / dt)
    
    x = np.linspace(0, L, N+1)
    T = np.ones(N+1) * T_init
    T[0]  = T_gauche
    T[-1] = T_droite
    
    save_step = max(1, nt // save_frac)
    T_save, t_save = [T.copy()], [0.0]

    T_new = T.copy()

    for n in range(1, nt+1):
        # Mise à jour des noeuds intérieurs (i = 1 à N-1)
        for i in range(1, N):
            T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
        # Mise à jour
        T = T_new.copy()
        # Sauvegarde périodique
        if n % save_step == 0:
            T_save.append(T.copy())
            t_save.append(n * dt)

    return x, np.array(t_save), np.array(T_save)


def simulation_chaleur_vect(L, alpha, N, r, t_fin, T_gauche, T_droite, T_init, save_frac=10):
    """
    Résout l'équation de la chaleur 1D par schéma FTCS explicite (version vectorisée).
    
    Paramètres :
    ------------
    L          : longueur de la barre (m)
    alpha      : diffusivité thermique (m²/s)
    N          : nombre d'intervalles spatiaux
    r          : nombre de Fourier discret (r <= 0.5 pour la stabilité)
    t_fin      : durée totale de simulation (s)
    T_gauche   : température à x=0 (°C)
    T_droite   : température à x=L (°C)
    T_init     : température initiale uniforme (°C)
    save_frac  : fraction de sauvegarde
    
    Retourne :
    ----------
    x, t_save, T_save
    """
    dx = L / N
    dt = r * dx**2 / alpha
    nt = int(t_fin / dt)
    
    x = np.linspace(0, L, N+1)
    T = np.ones(N+1) * T_init
    T[0]  = T_gauche
    T[-1] = T_droite
    
    save_every = max(1, nt // save_frac)
    T_save, t_save = [T.copy()], [0.0]
    
    for n in range(nt):
        # ---- Version vectorisée : une seule ligne ! ----
        T[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        
        # Les conditions aux limites sont automatiquement préservées
        # car on ne touche pas T[0] et T[-1]
        
        if (n+1) % save_every == 0:
            T_save.append(T.copy())
            t_save.append((n+1) * dt)
    
    return x, np.array(t_save), np.array(T_save)


def tracer_T_x(x, T, t, title="Évolution du profil de température dans la barre", show=True, ax=None):
    """
    Trace l'évolution du profil de température de la barre.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    colors = cm.plasma(np.linspace(0, 1, len(T)))

    for k, (T_k, t_k) in enumerate(zip(T, t)):
        ax.plot(x, T_k, color=colors[k], label=f't = {t_k:.0f} s')

    ax.set_xlabel('Position x (m)', fontsize=12)
    ax.set_ylabel('Température T (°C)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if show:
        plt.tight_layout()
        plt.show()

    return ax

def comparer_r_stabilite():
    """
    Compare l'effet du paramètre de stabilité r sur la solution numérique.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    r_values = [0.4, 0.5, 0.6]
    titres = ['r = 0.4 (Stable)', 'r = 0.5 (Limite)', 'r = 0.6 (INSTABLE !)']

    for ax, r_val, titre in zip(axes, r_values, titres):
        x, t_sv, T_sv = simulation_chaleur_vect(L=1.0, alpha=1.11e-4, N=20, r=r_val,
                                            t_fin=200, T_gauche=100, T_droite=0, T_init=0, save_frac=5)
        colors = cm.viridis(np.linspace(0, 1, len(T_sv)))
        for k in range(len(T_sv)):
            ax.plot(x, T_sv[k], color=colors[k], label=f't = {t_sv[k]:.0f} s')
        
        ax.set_title(titre, fontsize=11, fontweight='bold')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('T (°C)')
        ax.set_ylim(0, 100)   # Fixer les axes pour la comparaison
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.suptitle("Effet du paramètre de stabilité r sur la solution numérique", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def comparer_N_raffinement():
    """
    Compare les solutions pour différentes valeurs de N (raffinement spatial).
    """
    N_values = [5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(9, 6))

    for N_val in N_values:
        x_n, t_sv, T_sv = simulation_chaleur_vect(L=1.0, alpha=1.11e-5, 
                                                  N=N_val, r=0.4,
                                              t_fin=1500, T_gauche=100, T_droite=0, 
                                              T_init=0, save_frac=10)
        # On trace uniquement le profil final (état stationnaire)
        ax.plot(x_n, T_sv[-1], marker='o', markersize=4, label=f'N = {N_val}')
        
    ax.set_xlabel('Position x (m)', fontsize=12)
    ax.set_ylabel('Température T (°C)', fontsize=12)
    ax.set_title("Profil de température à t_fin pour différents raffinements", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # --- Paramètres physiques ---
    L     = 1.0          # Longueur de la barre (m)
    alpha = 1.11e-4      # Diffusivité thermique du cuivre (m²/s)
    T_gauche = 100.0     # Température à x=0 (°C)
    T_droite = 0.0       # Température à x=L (°C)
    T_init   = 0.0       # Température initiale (°C)

    # --- Paramètres numériques ---
    N  = 20              # Nombre d'intervalles spatiaux
    r = 0.4              # Nombre de Fourier discret
    t_fin = 3000         # Durée de simulation (s)
    
    print("=== Simulation de l'Équation de la Chaleur 1D ===")
    
    # 1. Comparaison des performances
    print("\n--- Comparaison des temps de calcul (N=40) ---")
    N_perf = 40
    
    debut = time.time()
    x, t_save, T_save = simulation_chaleur(L, alpha, N_perf, r, t_fin, T_gauche, T_droite, T_init)
    fin = time.time()
    print(f"Temps avec boucle for : {fin - debut:.3f} s")
    
    debut = time.time()
    x_vect, t_save_vect, T_save_vect = simulation_chaleur_vect(L, alpha, N_perf, r, t_fin, T_gauche, T_droite, T_init)
    fin = time.time()
    print(f"Temps version vectorisée : {fin - debut:.3f} s")
    
    # 2. Visualisation du profil standard
    print("\n--- Visualisation du profil (r=0.4, N=20) ---")
    x, t_save, T_save = simulation_chaleur_vect(L, alpha, N, r, t_fin, T_gauche, T_droite, T_init)
    tracer_T_x(x, T_save, t_save)
    
    # 3. Comparaison de la stabilité
    print("\n--- Influence du paramètre de stabilité r ---")
    comparer_r_stabilite()
    
    # 4. Comparaison du raffinement spatial
    print("\n--- Influence du raffinement spatial N ---")
    comparer_N_raffinement()

if __name__ == "__main__":
    main()
