import random
import pickle
from collections import defaultdict

# ----------------------------
# Parámetros del entorno
# ----------------------------
NUM_HEAPS = 3
MAX_HEAP_SIZE = 5
EPISODES = 30000
EPSILON_START = 1
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.99

# ----------------------------
# Funciones auxiliares
# ----------------------------
def get_possible_actions(state):
    actions = []
    for i, heap in enumerate(state):
        for amt in range(1, heap + 1):
            actions.append((i, amt))
    return actions

def apply_action(state, action):
    heap_idx, remove_amt = action
    state = list(state)
    state[heap_idx] -= remove_amt
    return tuple(state)

def is_terminal(state):
    return all(heap == 0 for heap in state)

def choose_action(state, Q, epsilon):
    actions = get_possible_actions(state)
    if not actions:
        return None
    if random.random() < epsilon:
        return random.choice(actions)
    q_vals = [Q[(state, a)] for a in actions]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
    return random.choice(best_actions)

def choose_best_action(state, Q):
    actions = get_possible_actions(state)
    q_vals = [Q.get((state, a), 0) for a in actions]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
    return random.choice(best_actions)

def xor_sum(state):
    result = 0
    for heap in state:
        result ^= heap
    return result

def print_state(state):
    print("\nEstado actual:")
    for i, heap in enumerate(state):
        print(f"  Montón {i + 1}: {'*' * heap} ({heap})")
    print()

# ----------------------------
# Entrenamiento Monte Carlo
# ----------------------------
Q = defaultdict(float)
returns_count = defaultdict(int)
epsilon = EPSILON_START

print("Entrenando al agente...")

for episode in range(EPISODES):
    state = tuple(random.randint(1, MAX_HEAP_SIZE) for _ in range(NUM_HEAPS))
    episode_history = []
    player_turn = random.randint(0, 1)

    while not is_terminal(state):
        if player_turn == 0:
            action = choose_action(state, Q, epsilon)
            if action is None:
                break
            episode_history.append((state, action))
            state = apply_action(state, action)
            if is_terminal(state):
                reward = 1
                break
        else:
            actions = get_possible_actions(state)
            xor_before = xor_sum(state)
            if xor_before == 0:
                action = random.choice(actions)
            else:
                good_actions = [a for a in actions if xor_sum(apply_action(state, a)) == 0]
                if good_actions:
                    action = random.choice(good_actions)
                else:
                    action = random.choice(actions)
            state = apply_action(state, action)
            if is_terminal(state):
                reward = -1
                break
        player_turn = 1 - player_turn
    else:
        reward = 0

    visited = set()
    for (s, a) in episode_history:
        if (s, a) not in visited:
            visited.add((s, a))
            returns_count[(s, a)] += 1
            Q[(s, a)] += (reward - Q[(s, a)]) / returns_count[(s, a)]

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

print("Entrenamiento completado.\n")

# Guardar el modelo entrenado
with open("modelo_nim.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

# ----------------------------
# Juego contra el agente
# ----------------------------
print("Iniciando una partida contra el agente entrenado...")

with open("modelo_nim.pkl", "rb") as f:
    Q = pickle.load(f)

state = tuple(random.randint(1, MAX_HEAP_SIZE) for _ in range(NUM_HEAPS))
player_turn = random.randint(0, 1)  # Turno aleatorio

while not is_terminal(state):
    print_state(state)
    if player_turn == 0:
        # Turno humano
        while True:
            try:
                heap_idx = int(input("Elige un montón (1-3): ")) - 1
                amt = int(input("¿Cuántos palos quitar?: "))
                if 0 <= heap_idx < NUM_HEAPS and 1 <= amt <= state[heap_idx]:
                    break
                else:
                    print("Movimiento inválido. Intenta de nuevo.")
            except ValueError:
                print("Entrada no válida. Intenta de nuevo.")
        state = apply_action(state, (heap_idx, amt))
        if is_terminal(state):
            print("\n¡Felicidades, ganaste!")
            break
    else:
        print("Turno del agente...")
        action = choose_best_action(state, Q)
        print(f"El agente quita {action[1]} del montón {action[0] + 1}")
        state = apply_action(state, action)
        if is_terminal(state):
            print("\nPerdiste. El agente ganó.")
            break
    player_turn = 1 - player_turn
