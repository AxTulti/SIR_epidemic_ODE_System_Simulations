import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt



# Case 1
D = 5
N = 160
radius = 0.6
recovery_rate = 0.5
recovery_rounds = 12

time = np.linspace(0, 5, 80)

def plot_SIR(df):
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
    theta = random.random() * 2 * np.pi
    r = np.sqrt(random.random()) * (D / 2)
    
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
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

def plot_current_state(df):
    X_s, Y_s = get_suceptible_coord(df)
    X_i, Y_i = get_infected_coord(df)
    X_r, Y_r = get_recovered_coord(df)

    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), D / 2, color='black', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    ax.set_aspect('equal')
    plt.scatter(X_i, Y_i, color = 'red')
    plt.scatter(X_s, Y_s, color = 'blue')
    plt.scatter(X_r, Y_r, color = 'green')
    plt.show()

def iterate_recovered(df):
    infecteds = df[df['state'] == 'I']
    
    for idx, infected in infecteds.iterrows():
        if infected.infected_rounds == recovery_rounds:
            df.loc[df['id'] == idx, 'state'] = 'R'
        else:
            df.loc[df['id'] == idx, 'infected_rounds'] += 1
    
def iterate_infection(df):
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
            
            if distance < radius: df.loc[df['id'] == idx_2, 'state'] = 'I'


df = pd.DataFrame(columns=['x', 'y', 'state', 'infected_rounds', 'id'])

for i in range(N):
    x, y = generate_position()
    df.loc[len(df)] = [ x, y, 'S', 0, i]
    
# Choose our Infected
chosen_one = random.randint(0, N-1)

df.loc[df['id'] == chosen_one, 'state'] = 'I'


SIR_df = pd.DataFrame(columns=['S', 'I', 'R'])

for t in time:
    SIR_df.loc[len(SIR_df)] = get_SIR_numbers(df)
    print(SIR_df)
    
    plot_current_state(df)
    iterate_infection(df)
    iterate_recovered(df)
    

plot_SIR(df)