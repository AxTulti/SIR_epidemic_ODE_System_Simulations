import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# Case 1
D = 12
N = 100
radius = 0.35
recovery_rate = 0.05
movement_rate = 0.8
infectance_rate = 0.35
fps = 50

time = np.linspace(0, 8, 112)

def get_normal_displacement():
    rad_movement = np.sqrt(np.abs(np.random.normal())) * movement_rate
    theta_movement = np.random.normal() * 2 * np.pi
    
    x = rad_movement * np.cos(theta_movement)
    y = rad_movement * np.sin(theta_movement)
    
    
    return (x, y)

def is_movement_out_of_bounds(x0, y0, x_displacement, y_displacement):
    vect_start = np.array([x0, y0])
    vect_displacement = np.array([x_displacement, y_displacement])
    vect_end = vect_start + vect_displacement
    
    is_valid = np.linalg.norm(vect_end) < (D / 2)
    
    return is_valid

def get_and_apply_displacements( df ):
    
    for idx, person in df.iterrows():
        is_displacement_valid = False
        x_d = 0
        y_d = 0
        
        while not is_displacement_valid:
            x_d, y_d = get_normal_displacement()
            x_0 = person.x
            y_0 = person.y
            is_displacement_valid = is_movement_out_of_bounds(x_0, y_0, x_d, y_d)
        
        df.loc[ df['id'] == idx, 'x' ] += x_d
        df.loc[ df['id'] == idx, 'y' ] += y_d

def plot_SIR(df):
    plt.clf()
    plt.ylabel('# of people')
    plt.xlabel('Time (days)')
    plt.plot(time, SIR_df['S'], color = 'blue', label = 'Suceptible' )
    plt.plot(time, SIR_df['I'], color = 'red', label = 'Infected')
    plt.plot(time, SIR_df['R'], color='green', label='Recovered')
    plt.legend()
    plt.show()

def get_SIR_numbers(df):
    suceptible = len(df[df['state'] == 'S'])
    infected = len(df[df['state'] == 'I'])
    recovered = len(df[df['state'] == 'R'])
    
    return (suceptible, infected, recovered)

def generate_position(x0, y0):
    position_valid = False
    
    while not position_valid:
        
        x = np.random.normal(loc = x0, scale = D / 20)
        y = np.random.normal(loc = y0, scale = D / 20)
        
        distance = np.linalg.norm([x, y])
        position_valid = distance <= D/2
    
    return (x, y)

def get_suceptible_coord(df):
    people = df[df['state'] == 'S']
    return (people.x, people.y)

def get_infected_coord(df):
    people = df[df['state'] == 'I']
    return (people.x, people.y)

def get_recovered_coord(df):
    people = df[df['state'] == 'R']
    return (people.x, people.y)

def plot_current_state(df, fig, ax):
    ax.clear()
    X_s, Y_s = get_suceptible_coord(df)
    X_i, Y_i = get_infected_coord(df)
    X_r, Y_r = get_recovered_coord(df)

    
    
    ax.scatter(X_i, Y_i, color = 'red')
    ax.scatter(X_s, Y_s, color = 'blue')
    ax.scatter(X_r, Y_r, color = 'green')
    
    # keep stable axis limits so frames don't jump
    ax.set_xlim(-D/2 - 0.5, D/2 + 0.5)
    ax.set_ylim(-D/2 - 0.5, D/2 + 0.5)

    # draw legend
    sus_patch = mpatches.Patch(color='blue', label='Susceptible')
    inf_patch = mpatches.Patch(color='red', label='Infected')
    rec_patch = mpatches.Patch(color='green', label='Recovered')
    
    ax.legend(handles=[sus_patch, inf_patch, rec_patch], loc='upper right')
    
    circle = mpatches.Circle((0, 0), D / 2, edgecolor='black', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    
    fig.canvas.draw()
    plt.pause( 1 / fps)


def iterate_recovered(df):
    infecteds = df[df['state'] == 'I']
    
    
    for idx, infected in infecteds.iterrows():
        recovery = random.random()
        if recovery < recovery_rate:
            df.loc[df['id'] == idx, 'state'] = 'R'
            
# def iterate_recovered(df):
#     infecteds = df[df['state'] == 'I']
    
#     for idx, infected in infecteds.iterrows():
#         if infected.infected_rounds == recovery_rounds:
#             df.loc[df['id'] == idx, 'state'] = 'R'
#         else:
#             df.loc[df['id'] == idx, 'infected_rounds'] += 1
    
def iterate_infection(df):
    infects_current = random.random() < infectance_rate
    
    infecteds = df[ df['state'] == 'I']
    suceptibles = df[ df['state'] == 'S']
    
    for idx, infected in infecteds.iterrows():
        x, y, state, inf_round, _id = infected
        infected_position = np.array([x, y])
        
        # get distances
        for idx_2, suceptible in suceptibles.iterrows():
            x, y, state, inf_round, _id = suceptible
            suceptible_position = np.array([x, y])
            
            distance = np.linalg.norm(infected_position - suceptible_position)
            
            if distance < radius and infects_current: df.loc[df['id'] == idx_2, 'state'] = 'I'


df = pd.DataFrame(columns=['x', 'y', 'state', 'infected_rounds', 'id'])

theta = random.random() * 2 * np.pi
r = np.sqrt(random.random()) * (D / 10)

x0 = np.cos(theta) * r
y0 = np.sin(theta) * r

for i in range(N):
    x, y = generate_position( x0, y0 )
    df.loc[len(df)] = [ x, y, 'S', 0, i]
    
# Choose our Infected
chosen_one = random.randint(0, N-1)

df.loc[df['id'] == chosen_one, 'state'] = 'I'


SIR_df = pd.DataFrame(columns=['S', 'I', 'R'])
fig, ax = plt.subplots()


for t in time:
    SIR_df.loc[len(SIR_df)] = get_SIR_numbers(df)

    ax.set_aspect('equal')
    plot_current_state(df, fig, ax)
    iterate_infection(df)
    iterate_recovered(df)
    get_and_apply_displacements(df)
    

plot_SIR(df)