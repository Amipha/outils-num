import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def simulation_chaleur(L, alpha, N, r, t_fin, T_gauche, T_droite, T_init, save_frac=10) :
    dx = L / N
    dt = r*dx**2/alpha
    nt = int ( t_fin / dt )
    # - - - Creation de la grille spatiale - - -
    x = np.linspace(0, L, N+1) # N + 1 points de x = 0 a x = L
    # - - - Condition initiale - - -
    T = np.ones (N + 1) * T_init

    T[0] = T_gauche
    T[-1] = T_droite

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

def Tracer_T_x(x, t, T):
    fig , ax = plt.subplots(figsize = (9 , 6) )
    colors = cm.plasma(np.linspace(0 , 1 , len(T)))

    for k, (T_k, t_k) in enumerate(zip(T, t)):
        ax.plot (x , T_k , color=colors[k], label=f"t = {t_k:.0f} s")

    ax.set_xlabel (  "Position x ( m )" )
    ax.set_ylabel ( "Temperature T ( °C )" )
    ax.legend(loc='upper right')
    ax.set_title ( "Evolution du profil de température de la barre" )
    ax.grid (True, linestyle = "--"  , alpha = 0.5)
    plt.tight_layout()
    plt.show()

def simulation_chaleur_vect(L, alpha, N, r, t_fin, T_gauche, T_droite , T_init , save_step_frac = 10) :
    """
    Resout l ’ equation de la chaleur 1 D par schema FTCS
    ( version vectorisee NumPy ) .
    Parametres
    ----------
    L: longueur de la barre ( m )
    alpha: diffusivite thermique ( m ^2 / s )
    N: nombre d ’ intervalles spatiaux
    r: nombre de Fourier discret ( r <= 0.5)
    t_fin: duree totale de simulation ( s )
    T_gauche: temperature a x = 0 ( degC )
    T_droite: temperature a x = L ( degC )
    f_init: fonction d'initialisation de la tempéraute f_init(x)
    save_frac  fraction de pas de temps a sauvegarder
    Retourne
    --------
    x , t_save , T_save
    """
    dx = L / N
    dt = r * dx ** 2 / alpha
    nt = int ( t_fin / dt )
    x = np.linspace (0 , L , N + 1)
    T = np.ones ( N + 1) * T_init
    T [0] = T_gauche
    T [-1] = T_droite
    save_every = max(1, nt // save_step_frac)
    T_save , t_save = [ T.copy()] , [0.0]
    for n in range (nt) :
    # Version vectorisee : une seule ligne !
        T[1: - 1] = T[1: - 1] + r * ( T[2:] - 2 * T[1: - 1] + T[: - 2])
    # T [0] et T [ - 1] ne sont pas modifies ( CL preservees )
        if ( n + 1) % save_every == 0:
            T_save.append ( T.copy() )
            t_save.append (( n + 1) * dt )
    return x , np . array ( t_save ) , np . array ( T_save )

def simulation_chaleur_vect_f(L, alpha, N, r, t_fin, T_gauche, T_droite , f_init , save_step_frac = 10) :
    """
    Resout l ’ equation de la chaleur 1 D par schema FTCS
    ( version vectorisee NumPy ) .
    Parametres
    ----------
    L: longueur de la barre ( m )
    alpha: diffusivite thermique ( m ^2 / s )
    N: nombre d ’ intervalles spatiaux
    r: nombre de Fourier discret ( r <= 0.5)
    t_fin: duree totale de simulation ( s )
    T_gauche: temperature a x = 0 ( degC )
    T_droite: temperature a x = L ( degC )
    T_init: temperature initiale uniforme ( degC )
    save_frac  fraction de pas de temps a sauvegarder
    Retourne
    --------
    x , t_save , T_save
    """
    dx = L / N
    dt = r * dx ** 2 / alpha
    nt = int ( t_fin / dt )
    x = np.linspace (0 , L , N + 1)
    T = f_init(x)
    T [0] = T_gauche
    T [-1] = T_droite
    save_every = max(1, nt // save_step_frac)
    T_save , t_save = [ T.copy()] , [0.0]
    for n in range (nt) :
    # Version vectorisee : une seule ligne !
        T[1: - 1] = T[1: - 1] + r * ( T[2:] - 2 * T[1: - 1] + T[: - 2])
    # T [0] et T [ - 1] ne sont pas modifies ( CL preservees )
        if ( n + 1) % save_every == 0:
            T_save.append ( T.copy() )
            t_save.append (( n + 1) * dt )
    return x , np . array ( t_save ) , np . array ( T_save )