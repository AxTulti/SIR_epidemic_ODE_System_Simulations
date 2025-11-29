import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Case 1
D = 5
N = 160
radius = 0.35
recovery_rate = 0.05
movement_rate = 0.3
infectance_rate = 0.6
fps = 20

time = np.linspace(0, 8, 112)

def get_normal_displacement():
    rad_movement = np.sqrt(np.abs(np.random.normal())) * movement_rate
    theta_movement = np.random.normal() * 2 * np.pi
    
    x = rad_movement * np.cos(theta_movement)
    y = rad_movement * np.sin(theta_movement)
    
    return (x, y)

def is_movement_out_of_bounds(x0, y0, x_displacement, y_displacement):
    x_valid = 0 <= x0 + x_displacement <= D
    y_valid = 0 <= y0 + y_displacement <= D
    
    return x_valid and y_valid

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

def generate_position():
    x = random.random() * D
    y = random.random() * D
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


def plot_animation_between_two_states(previous_df, current_df, fig, ax):
    ax.clear()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal')

    trajectories = []
    colors = []

    for idx, row_prev in previous_df.iterrows():
        x0, y0 = row_prev.x, row_prev.y
        row_curr = current_df.loc[idx]
        xf, yf = row_curr.x, row_curr.y

        dx, dy = xf - x0, yf - y0
        traj = lambda t, x0=x0, y0=y0, dx=dx, dy=dy: np.array([x0 + t*dx, y0 + t*dy])
        trajectories.append(traj)

        # acc to curr state
        state = row_curr.state
        if state == 'S': colors.append("blue")
        elif state == 'I': colors.append("red")
        else: colors.append("green")

    # Animate
    for tau in np.linspace(0, 1, 20):
        for scatter in ax.collections:
            scatter.remove()     # del oly scater

        pts = np.array([traj(tau) for traj in trajectories])
        ax.scatter(pts[:,0], pts[:,1], c=colors)

        plt.pause(0.001)
    

def plot_current_state(df, fig, ax):
    ax.clear()
    X_s, Y_s = get_suceptible_coord(df)
    X_i, Y_i = get_infected_coord(df)
    X_r, Y_r = get_recovered_coord(df)
    
    ax.set_xlim(0,D)
    ax.set_ylim(0, D)

    
    
    ax.scatter(X_i, Y_i, color = 'red')
    ax.scatter(X_s, Y_s, color = 'blue')
    ax.scatter(X_r, Y_r, color = 'green')

    # draw legend
    sus_patch = mpatches.Patch(color='blue', label='Susceptible')
    inf_patch = mpatches.Patch(color='red', label='Infected')
    rec_patch = mpatches.Patch(color='green', label='Recovered')
    
    ax.legend(handles=[sus_patch, inf_patch, rec_patch], loc='upper right')

    ax.set_aspect('equal')
    
    fig.canvas.draw()
    plt.pause(1/fps)

# def plot_current_state(df):
#     X_s, Y_s = get_suceptible_coord(df)
#     X_i, Y_i = get_infected_coord(df)
#     X_r, Y_r = get_recovered_coord(df)

#     plt.scatter(X_i, Y_i, color = 'red')
#     plt.scatter(X_s, Y_s, color = 'blue')
#     plt.scatter(X_r, Y_r, color = 'green')
#     plt.show()

def iterate_recovered(df):
    infecteds = df[df['state'] == 'I']
    
    
    for idx, infected in infecteds.iterrows():
        recovery = random.random()
        if recovery < recovery_rate:
            df.loc[df['id'] == idx, 'state'] = 'R'

    
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

for i in range(N):
    x, y = generate_position()
    df.loc[len(df)] = [ x, y, 'S', 0, i]
    
# Choose our Infected
chosen_one = random.randint(0, N-1)

df.loc[df['id'] == chosen_one, 'state'] = 'I'


SIR_df = pd.DataFrame(columns=['S', 'I', 'R'])

fig, ax = plt.subplots()

for t in time:
    SIR_df.loc[len(SIR_df)] = get_SIR_numbers(df)
    last_df = df.copy()
    iterate_infection(df)
    iterate_recovered(df)
    get_and_apply_displacements(df)
    
    plot_animation_between_two_states(last_df, df, fig, ax)
    
    
    
    

plot_SIR(df)