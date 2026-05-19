import json

cells = []

def md(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

md(r"""# TP : Résolution Numérique de l'Équation de la Chaleur 1D
## Simulation de la Diffusion Thermique dans une Barre Métallique

**Durée :** 4 heures | **Outils :** Python 3, NumPy, Matplotlib
**Thèmes :** EDP, Différences finies, Stabilité

Ce notebook constitue le rendu complet du TP. L'ensemble des questions posées y sont traitées.

---
## Partie 1 : Rappels Théoriques

L'évolution de la température $T(x,t)$ dans une barre 1D est régie par l'équation de la chaleur :
$$ \frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} $$
Par la méthode des **différences finies**, on aboutit au schéma d'Euler explicite (FTCS) :
$$ T_i^{n+1} = T_i^n + r \left(T_{i+1}^n - 2T_i^n + T_{i-1}^n\right) \quad \text{avec} \quad r = \frac{\alpha \Delta t}{\Delta x^2} $$
Pour que ce schéma soit stable, il faut respecter la **condition de stabilité** : $r \le 0.5$.

---
## Partie 2 : Mise en Place du Code de Base

### 2.1 Paramètres de Simulation""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# --- Paramètres physiques ---
L = 1.0              # Longueur de la barre (m)
alpha = 1.11e-4      # Diffusivité thermique du cuivre (m^2/s)
T_gauche = 100.0     # Température à x=0 (°C)
T_droite = 0.0       # Température à x=L (°C)
T_init = 0.0         # Température initiale uniforme (°C)

# --- Paramètres numériques ---
N = 20               # Nombre d'intervalles spatiaux
dx = L / N           # Pas spatial (m)
r = 0.4              # Nombre de Fourier discret (doit être <= 0.5)

# Calcul du pas de temps dt d'après la formule r = alpha*dt / dx^2
dt = r * (dx**2) / alpha

t_fin = 3000         # Durée de simulation (s)
nt = int(t_fin / dt)

print(f"dx = {dx:.4f} m")
print(f"dt = {dt:.4f} s")
print(f"r  = {r:.4f} (doit être <= 0.5)")
print(f"Nombre de pas de temps : {nt}")
""")

md(r"""### Réponses 2.1 :
1. **Calcul de $\Delta t$** : La formule de stabilité donne $r = \frac{\alpha \Delta t}{\Delta x^2}$, on isole donc $\Delta t = \frac{r \Delta x^2}{\alpha}$.
2. **Condition de stabilité** : Avec $r=0.4$, nous sommes bien dans le domaine de stabilité ($r \le 0.5$).
3. **Nombre de nœuds** : La barre est découpée en $N=20$ intervalles, soit $N+1=21$ nœuds au total. Les deux extrémités étant imposées par les conditions aux limites, il y a **19 nœuds intérieurs** dont la température est inconnue et doit être calculée à chaque pas de temps.

### 2.2 Initialisation du Domaine""")

code("""# --- Création de la grille spatiale ---
x = np.linspace(0, L, N+1)   # N+1 points de x=0 à x=L

# --- Condition initiale ---
T = np.ones(N+1) * T_init

# --- Application des conditions aux limites ---
T[0] = T_gauche
T[-1] = T_droite

# --- Visualisation du profil initial ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, T, 'o-', color='royalblue', linewidth=2, markersize=6)
ax.set_xlabel('Position x (m)')
ax.set_ylabel('Température T (°C)')
ax.set_title("Profil initial de température T(x, t=0)")
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
""")

md("""### 2.3 Boucle Temporelle Principale""")

code("""# --- Stockage périodique de la solution ---
save_frac = 10
save_step = max(1, nt // save_frac)
T_save = [T.copy()]
t_save = [0.0]

T_new = T.copy()

for n in range(1, nt+1):
    # Mise à jour des noeuds intérieurs (i = 1 à N-1)
    for i in range(1, N):
        T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
        
    T = T_new.copy() 
    
    # Sauvegarde périodique
    if n % save_step == 0:
        T_save.append(T.copy()) 
        t_save.append(n * dt)

T_save = np.array(T_save)
print(f"Nombre de profils sauvegardés : {len(T_save)}")
""")

md("""### Réponses 2.3 :
- **À quoi sert `T_new` ?** Lors du calcul itératif spatial dans la boucle, le calcul pour l'indice `i` dépend des valeurs aux indices `i-1`, `i` et `i+1` à l'instant précédent. Si nous utilisions un seul tableau `T` pour lire et écrire, l'évaluation au nœud `i+1` utiliserait une valeur `T[i]` déjà modifiée, ce qui fausserait complètement le schéma mathématique. On stocke donc les nouvelles valeurs temporairement dans `T_new`.
- **Importance de `.copy()`** : En Python, les variables contenant des tableaux NumPy agissent comme des références (pointeurs). Si l'on fait `T_save.append(T)`, on sauvegarde le même objet mémoire à chaque fois. À la fin, `T_save` contiendrait 10 fois le même (dernier) profil ! `.copy()` force la création d'un tableau distinct en mémoire.

### 2.4 Visualisation des Résultats""")

code("""fig, ax = plt.subplots(figsize=(9, 6))

colors = cm.plasma(np.linspace(0, 1, len(T_save)))

for k, (T_k, t_k) in enumerate(zip(T_save, t_save)):
    ax.plot(x, T_k, color=colors[k], label=f't = {t_k:.0f} s')

ax.set_xlabel('Position x (m)')
ax.set_ylabel('Température T (°C)')
ax.set_title('Évolution du profil de température dans la barre')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
""")

md(r"""### Réponses 2.4 :
- **Description qualitative** : Le profil initial présente un gradient infini au bord chaud. Avec le temps, la chaleur diffuse progressivement vers la droite. Les gradients de température s'atténuent et le profil s'arrondit.
- **Convergence en l'infini** : Le système converge vers un état stationnaire où la température ne dépend plus du temps. Le profil devient une droite parfaite reliant 100°C à gauche et 0°C à droite. Physiquement, le flux de chaleur devient constant à l'intérieur du matériau et plus aucune zone ne "stocke" d'énergie (loi de Fourier).
- **Profil théorique à l'état stationnaire** : En état stationnaire, $\frac{\partial T}{\partial t} = 0$, donc l'équation devient $\frac{d^2T}{dx^2} = 0$. La solution est de la forme $T(x) = Ax + B$. Avec les conditions $T(0) = 100$ et $T(L) = 0$, on trouve très facilement $B=100$ et $A=-100/L$. Donc $T_{eq}(x) = 100(1 - x/L)$.

### 2.5 Encapsulation dans des fonctions""")

code("""def tracer_T_x(x, T_list, t_list, title="Évolution du profil de température"):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = cm.plasma(np.linspace(0, 1, len(T_list)))
    
    for k, (T_k, t_k) in enumerate(zip(T_list, t_list)):
        ax.plot(x, T_k, color=colors[k], label=f't = {t_k:.0f} s')
        
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Température T (°C)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def simulation_chaleur(L, alpha, N, r, t_fin, T_gauche, T_droite, T_init, save_frac=10):
    dx = L / N
    dt = r * dx**2 / alpha
    nt = int(t_fin / dt)
    
    x = np.linspace(0, L, N+1)
    T = np.ones(N+1) * T_init
    T[0] = T_gauche
    T[-1] = T_droite
    
    save_step = max(1, nt // save_frac)
    T_save, t_save = [T.copy()], [0.0]
    
    T_new = T.copy()
    
    for n in range(1, nt+1):
        for i in range(1, N):
            T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
        T = T_new.copy()
        
        if n % save_step == 0:
            T_save.append(T.copy())
            t_save.append(n * dt)
            
    return x, np.array(t_save), np.array(T_save)
""")

md("""---
## Partie 3 : Optimisation avec NumPy

La double boucle Python est trop lente. Nous utilisons l'indexation de tableaux NumPy (Slicing) pour vectoriser l'opération.

### 3.1 Version Vectorisée""")

code("""def simulation_chaleur_vect(L, alpha, N, r, t_fin, T_gauche, T_droite, T_init, save_frac=10):
    dx = L / N
    dt = r * dx**2 / alpha
    nt = int(t_fin / dt)
    
    x = np.linspace(0, L, N+1)
    T = np.ones(N+1) * T_init
    T[0] = T_gauche
    T[-1] = T_droite
    
    save_every = max(1, nt // save_frac)
    T_save, t_save = [T.copy()], [0.0]
    
    for n in range(nt):
        # Version vectorisée : on met à jour tous les noeuds intérieurs d'un coup
        T[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        
        if (n + 1) % save_every == 0:
            T_save.append(T.copy())
            t_save.append((n + 1) * dt)
            
    return x, np.array(t_save), np.array(T_save)

# Test de la nouvelle fonction
x_v, t_sv, T_sv = simulation_chaleur_vect(L=1.0, alpha=1.11e-4, N=20, r=0.4, t_fin=3000, T_gauche=100, T_droite=0, T_init=0)
tracer_T_x(x_v, T_sv, t_sv, title="Évolution du profil (Version vectorisée)")
""")

md("""### Explication de l'instruction vectorisée :
`T[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])`
- `T[1:-1]` désigne le sous-tableau constitué des **nœuds intérieurs** (ceux à mettre à jour, excluant les bords à l'indice 0 et au dernier indice).
- `T[2:]` correspond aux **nœuds situés à droite** de chaque nœud intérieur.
- `T[:-2]` correspond aux **nœuds situés à gauche** de chaque nœud intérieur.
- L'opération algébrique s'effectue terme à terme très rapidement en C grâce au moteur de NumPy, préservant par ailleurs nos conditions de Dirichlet sur les bords.

### 3.2 Comparaison des Temps de Calcul""")

code("""N_grand = 200

# Version avec boucle
debut = time.time()
_ = simulation_chaleur(L=1.0, alpha=1.11e-4, N=N_grand, r=0.4, t_fin=1000, T_gauche=100, T_droite=0, T_init=0, save_frac=1)
fin = time.time()
print(f"Temps boucle for : {fin - debut:.3f} s")

# Version vectorisée
debut = time.time()
_ = simulation_chaleur_vect(L=1.0, alpha=1.11e-4, N=N_grand, r=0.4, t_fin=1000, T_gauche=100, T_droite=0, T_init=0, save_frac=1)
fin = time.time()
print(f"Temps vectorisé : {fin - debut:.3f} s")
""")

md("""**Conclusion** : Le gain de temps de calcul de la vectorisation NumPy est énorme (l'exécution est souvent de 10 à 100 fois plus rapide !). La vectorisation est indispensable pour simuler des EDP à haute résolution.

---
## Partie 4 : Analyse de la Stabilité Numérique

### 4.1 Influence du Paramètre r""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 5))
r_values = [0.4, 0.5, 0.6]
titres = ['r = 0.4 (Stable)', 'r = 0.5 (Limite)', 'r = 0.6 (INSTABLE !)']

for ax, r_val, titre in zip(axes, r_values, titres):
    x_r, t_sv_r, T_sv_r = simulation_chaleur_vect(L=1.0, alpha=1.11e-4, N=20, r=r_val, t_fin=500, T_gauche=100, T_droite=0, T_init=0, save_frac=5)
    colors = cm.viridis(np.linspace(0, 1, len(T_sv_r)))
    
    for k in range(len(T_sv_r)):
        ax.plot(x_r, T_sv_r[k], color=colors[k])
        
    ax.set_title(titre)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('T (°C)')
    # Fixer les ordonnées pour que la divergence soit flagrante
    if r_val == 0.6:
        ax.set_ylim(-50, 150)
    else:
        ax.set_ylim(0, 100)
    
plt.tight_layout()
plt.show()
""")

md(r"""### Réponses 4.1 :
- **Observation** : Pour $r=0.4$ la convergence est douce et lisse. Pour $r=0.5$ on observe une évolution stable mais limitrophe. Pour $r=0.6$, le profil entre en violente résonance, la courbe produit des zig-zags très amples (valeurs négatives ou au-delà de 100°C) qui s'amplifient.
- **Physiquement** : Pour $r > 0.5$, le transfert de chaleur calculé sortant d'une cellule donnée (par rapport au temps alloué $\Delta t$) est tellement grand qu'il vide totalement la cellule thermique et la fait passer en deça de la température de ses voisines. C'est l'apparition de températures "négatives" aberrantes, créant une suroscillation qui va et vient. **Ce comportement n'a aucun sens physique.**
- **Choix en pratique** : On impose en général le pas d'espace $\Delta x$ pour la résolution voulue. Ensuite, on choisit une valeur de sécurité stricte pour le paramètre de Fourier, ex: $r=0.4$ (pour être sous 0.5) et on déduit le pas de temps maximum sûr par $\Delta t = 0.4 \cdot \frac{\Delta x^2}{\alpha}$.

### 4.2 Influence du Raffinement Spatial""")

code("""N_values = [5, 10, 20, 50]
fig, ax = plt.subplots(figsize=(9, 6))

for n_val in N_values:
    # On simule jusqu'à un temps arbitraire fixé (ex t_fin = 500s)
    x_n, t_sv_n, T_sv_n = simulation_chaleur_vect(L=1.0, alpha=1.11e-4, N=n_val, r=0.4, t_fin=500, T_gauche=100, T_droite=0, T_init=0, save_frac=1)
    # On trace uniquement le dernier profil
    ax.plot(x_n, T_sv_n[-1], marker='o', markersize=4, label=f'N = {n_val}')

ax.set_xlabel('Position x (m)')
ax.set_ylabel('Température T (°C)')
ax.set_title('Profil de température à t = 500s pour différents raffinements')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()
""")

md("""**Observation** : À mesure que N augmente, la résolution s'affine. Les profils pour N=20 et N=50 sont presque superposés, indiquant que la méthode numérique **converge**.

---
## Partie 5 : Cas Physiques Plus Réalistes

### 5.1 & 5.2 Externalisation et Condition Initiale en "Pic" Gaussien""")

code("""def simulation_chaleur_vect_finit(L, alpha, N, r, t_fin, T_gauche, T_droite, f_init, save_frac=10):
    dx = L / N
    dt = r * dx**2 / alpha
    nt = int(t_fin / dt)
    
    x = np.linspace(0, L, N+1)
    
    # Appel de la fonction d'initialisation (fonction extérieure)
    T = f_init(x)
    T[0] = T_gauche
    T[-1] = T_droite
    
    save_every = max(1, nt // save_frac)
    T_save, t_save = [T.copy()], [0.0]
    
    for n in range(nt):
        T[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        if (n + 1) % save_every == 0:
            T_save.append(T.copy())
            t_save.append((n + 1) * dt)
            
    return x, np.array(t_save), np.array(T_save)

def f_init_gauss(x):
    sigma = 0.05
    Tmax = 100
    L = x[-1]
    return Tmax * np.exp(-((x - L/2)**2) / (2 * sigma**2))

def f_init_gauss_fin(x):
    sigma = 0.02
    Tmax = 100
    L = x[-1]
    return Tmax * np.exp(-((x - L/2)**2) / (2 * sigma**2))

# Simulation pour la gaussienne normale (sigma=0.05)
x_p, t_p, T_p = simulation_chaleur_vect_finit(L=1.0, alpha=1.11e-4, N=100, r=0.4, t_fin=500, T_gauche=0, T_droite=0, f_init=f_init_gauss, save_frac=10)
tracer_T_x(x_p, T_p, t_p, title="Évolution du profil Gaussien (sigma=0.05)")

# Simulation pour la gaussienne fine (sigma=0.02)
x_pf, t_pf, T_pf = simulation_chaleur_vect_finit(L=1.0, alpha=1.11e-4, N=100, r=0.4, t_fin=500, T_gauche=0, T_droite=0, f_init=f_init_gauss_fin, save_frac=10)
tracer_T_x(x_pf, T_pf, t_pf, title="Évolution du profil Gaussien fin (sigma=0.02)")
""")

md(r"""### Réponses 5.2 :
- **Évolution du pic** : L'énergie thermique concentrée au milieu se dissipe vers la gauche et vers la droite simultanément. Le pic s'affaisse et s'élargit.
- **Convergence** : Lorsque $t \to +\infty$, le système va converger vers une température globale de 0°C. Puisque les bords sont forcés à 0°C (Dirichlet), la chaleur finit entièrement perdue dans ces puits thermiques.
- **Avec $\sigma = 0.02$** : Le pic très fin engendre un gradient thermique spatial $\frac{\partial^2T}{\partial x^2}$ beaucoup plus fort. La diminution d'amplitude de la température au centre se fait très rapidement sur les tout premiers pas de temps comparativement au pic plus large.

### 5.3 Carte de Chaleur Spatio-Temporelle (Heatmap)""")

code("""def f_init_zero(x):
    return np.zeros_like(x)

# Simulation cas nominal pour la Heatmap
x_hm, t_map, T_map = simulation_chaleur_vect_finit(L=1.0, alpha=1.11e-4, N=100, r=0.4, t_fin=3000, T_gauche=100, T_droite=0, f_init=f_init_zero, save_frac=50)

fig, ax = plt.subplots(figsize=(10, 6))
# np.meshgrid permet de créer des grilles 2D adaptées pour contourf
X, Y = np.meshgrid(x_hm, t_map)
c = ax.contourf(X, Y, T_map, levels=50, cmap='hot')
fig.colorbar(c, ax=ax, label='Température (°C)')
ax.set_xlabel('Position x (m)', fontsize=12)
ax.set_ylabel('Temps t (s)', fontsize=12)
ax.set_title("Carte spatio-temporelle de la température T(x,t)", fontsize=13)
plt.tight_layout()
plt.show()

# Heatmap pour le cas Pic (sigma=0.05)
x_hm2, t_map2, T_map2 = simulation_chaleur_vect_finit(L=1.0, alpha=1.11e-4, N=100, r=0.4, t_fin=500, T_gauche=0, T_droite=0, f_init=f_init_gauss, save_frac=50)

fig, ax = plt.subplots(figsize=(10, 6))
X2, Y2 = np.meshgrid(x_hm2, t_map2)
c2 = ax.contourf(X2, Y2, T_map2, levels=50, cmap='hot')
fig.colorbar(c2, ax=ax, label='Température (°C)')
ax.set_xlabel('Position x (m)', fontsize=12)
ax.set_ylabel('Temps t (s)', fontsize=12)
ax.set_title("Carte spatio-temporelle - Cas Pic", fontsize=13)
plt.tight_layout()
plt.show()
""")

md("""### Réponses 5.3 :
- **Interprétation** : L'axe X représente la position, l'axe Y le temps. Les couleurs correspondent au profil thermique au fil de l'évolution temporelle.
- **Diffusion rapide** : Elle a lieu dans les premiers instants (bas du graphe) près du bord gauche pour le premier cas, et au centre pour le cas pic. C'est là que les isocourbes sont obliques (variations rapides).
- **À grande valeur de t** : Les isocourbes deviennent de simples bandes verticales, illustrant visuellement l'atteinte de l'état stationnaire (les zones colorées ne bougent plus verticalement).

### 5.4 Condition Adiabatique""")

code("""def simulation_chaleur_adiabatique(L, alpha, N, r, t_fin, T_gauche, f_init, save_frac=10):
    dx = L / N
    dt = r * dx**2 / alpha
    nt = int(t_fin / dt)
    
    x = np.linspace(0, L, N+1)
    T = f_init(x)
    T[0] = T_gauche
    
    save_every = max(1, nt // save_frac)
    T_save, t_save = [T.copy()], [0.0]
    
    for n in range(nt):
        T[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        # Imposition de la condition adiabatique: flux nul à x=L -> T(N) = T(N-1)
        T[-1] = T[-2]
        
        if (n + 1) % save_every == 0:
            T_save.append(T.copy())
            t_save.append((n + 1) * dt)
            
    return x, np.array(t_save), np.array(T_save)

x_ad, t_ad, T_ad = simulation_chaleur_adiabatique(L=1.0, alpha=1.11e-4, N=50, r=0.4, t_fin=8000, T_gauche=100, f_init=f_init_zero, save_frac=10)
tracer_T_x(x_ad, T_ad, t_ad, title="Évolution avec Bord Droit Adiabatique")
""")

md("""**Observation Condition Adiabatique** : Le bord droit étant isolé (flux nul $T_N = T_{N-1}$), la chaleur apportée par la gauche s'accumule indéfiniment. Le profil tend in-fine vers une constante (100°C partout). On remarque que les profils de température croisent le bord droit avec une tangente horizontale, ce qui illustre un gradient de chaleur nul à cet endroit.

---
## Partie 6 : Solution Analytique et Validation

### 6.1 et 6.2 Solution Analytique, Implémentation et Comparaison""")

code("""def solution_analytique(x, t, L, alpha, T_gauche, T_droite, T_init, N_termes=50):
    # Partie stationnaire
    T_st = T_gauche + (T_droite - T_gauche) * x / L
    
    # Partie transitoire (série de Fourier)
    T_trans = np.zeros_like(x)
    for n in range(1, N_termes + 1):
        Cn = (2 / (n * np.pi)) * ((T_init - T_gauche) - ((-1)**n) * (T_init - T_droite))
        T_trans += Cn * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t)
        
    return T_st + T_trans

# Comparaison Numérique vs Analytique
x_num, t_sv, T_sv_num = simulation_chaleur_vect_finit(L=1.0, alpha=1.11e-4, N=20, r=0.4, t_fin=3000, T_gauche=100, T_droite=0, f_init=f_init_zero, save_frac=100)

n_sv = len(t_sv)
n_compare = [int(n_sv/12), int(n_sv/4), int(n_sv/2), n_sv-1]
x_ana = np.linspace(0, 1.0, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, n in zip(axes, n_compare):
    T_ana = solution_analytique(x_ana, t_sv[n], L=1.0, alpha=1.11e-4, T_gauche=100, T_droite=0, T_init=0)
    
    ax.plot(x_ana, T_ana, 'k-', label='Analytique', linewidth=2)
    ax.plot(x_num, T_sv_num[n], 'ro', label='Numérique', markersize=5)
    
    ax.set_title(f't = {t_sv[n]:.1f} s')
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('T (°C)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Comparaison Numérique vs Analytique", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

md("""### 6.3 Calcul de l'Erreur""")

code("""def tracer_erreur(N_val):
    x_num, t_sv, T_sv_num = simulation_chaleur_vect_finit(L=1.0, alpha=1.11e-4, N=N_val, r=0.4, t_fin=3000, T_gauche=100, T_droite=0, f_init=f_init_zero, save_frac=100)
    
    erreurs = []
    # On évite le tout premier pas (t=0) à cause du front brutal
    for idx, t_val in enumerate(t_sv[1:]):
        T_ana_pts = solution_analytique(x_num, t_val, L=1.0, alpha=1.11e-4, T_gauche=100, T_droite=0, T_init=0)
        erreur = np.max(np.abs(T_sv_num[idx+1] - T_ana_pts))
        erreurs.append(erreur)
        
    plt.semilogy(t_sv[1:], erreurs, marker='o', markersize=3, label=f'N={N_val}')

fig, ax = plt.subplots(figsize=(8, 5))
tracer_erreur(10)
tracer_erreur(100)
ax.set_xlabel('Temps t (s)', fontsize=12)
ax.set_ylabel('Erreur max |T_num - T_ana| (°C)', fontsize=11)
ax.set_title("Évolution de l'erreur numérique au cours du temps", fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()
""")

md(r"""### Réponses 6.3 :
- **Accord** : Oui, les profils numériques se superposent quasiment parfaitement à la solution analytique, ce qui valide l'implémentation.
- **Évolution de l'erreur** : L'erreur maximale baisse très rapidement avec le temps. Physiquement, le choc thermique du début (100°C directement à côté de 0°C) implique un gradient localement infini que la grille discrète lisse (erreur forte initiale). L'équation de la chaleur "atténue" les hautes fréquences, donc à mesure que le profil s'adoucit, l'erreur de troncature de notre discrétisation spatiale diminue drastiquement.
- **Effet du raffinement** : Le passage de $N=10$ à $N=100$ fait baisser l'erreur maximale d'au moins un facteur 10. Augmenter le raffinement spatial réduit considérablement l'erreur de discrétisation de notre schéma centré en espace.
""")

notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('/home/kenan/Documents/Esiroi/esiroi/Outil num/TP/outils-num/TP1_Resolution_Chaleur_Complet.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook generated successfully!")
