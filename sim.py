import numpy as np

def simulation_chaleur(L, alpha, N, r, t_fin, T_gauch, T_droite, T_init, save_frac=10) :

    # - - - Creation de la grille spatiale - - -
    x = np.linspace(0, L, N+1) # N + 1 points de x = 0 a x = L
    # - - - Condition initiale - - -
    T = np.ones (N + 1) * T_init

    save_step = max(1 , nt//save_frac)
    T_save = [T.copy()]
    t_save = [0.0]
    T_new = T.copy()
    for n in range (1 , nt + 1) :
        for i in range (1 , N ) :
            T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
        T = T_new.copy()
        if n % save_step == 0:
            T_save.append(T.copy())
            t_save.append(n * dt)
    T_save = np.array ( T_save )
    print ( f" Nombre de profils sauvegardes : {len(T_save)} " )
    return x, np.array(t_save), np.array(T_save)