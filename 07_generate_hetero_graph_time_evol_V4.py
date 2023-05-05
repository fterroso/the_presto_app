#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch


# In[2]:


print(torch.__version__)


# In[3]:


from torch import Tensor
#from torch import nn
import os
import numpy as np
import pandas as pd
from ast import literal_eval
import itertools
from datetime import date, timedelta


# In[4]:


from tqdm.notebook import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


# In[5]:


time_window_size= 30 #days
init_date='2004-01-01'
final_date='2021-12-31'


# ### Read songs data

# In[6]:


tracks_info= pd.read_csv(os.path.join('data', 'generated', '05b_all_tracks_w_feat_and_genre.csv'),
                         converters={"artist_id": literal_eval, 'artist_name': literal_eval, 'genres':literal_eval}, 
                         parse_dates=['release_date'],
                         index_col=0)
tracks_info= tracks_info[(tracks_info['release_date']>init_date) & (tracks_info['release_date']<=final_date)]
tracks_info= tracks_info[tracks_info['num_genres']>0]

tracks_info= tracks_info.reset_index(drop=True)
tracks_info['num_artists']= tracks_info['artist_id'].apply(lambda x:len(x))
tracks_info


# In[7]:


tracks_info.info()


# In[8]:


tracks_info.isna().sum().sum()


# In[9]:


track_map= tracks_info['track_id'].to_dict()
track_map


# In[10]:


single_tracks= tracks_info[tracks_info['num_artists']==1]
collab_tracks= tracks_info[tracks_info['num_artists']>1]


# In[11]:


single_tracks.shape, collab_tracks.shape


# ### Read artists data

# In[12]:


artist_info_df= pd.read_csv(os.path.join('data','generated', '04_lfm_spotify_artist_info_V2.csv'), 
                         converters={"genres": literal_eval}, index_col=0)
artist_info_df= artist_info_df.dropna(axis=0)
artist_info_df= artist_info_df.reset_index(drop=True)
artist_info_df


# In[13]:


artists_map= artist_info_df['artist_code'].to_dict()


# In[14]:


artist_info_df.shape


# In[15]:


artist_info_df.corr()


# In[16]:


artist_info_df['followers_norm']=(artist_info_df['followers']-artist_info_df['followers'].mean())/artist_info_df['followers'].std()
artist_info_df['popularity_norm']=(artist_info_df['popularity']-artist_info_df['popularity'].mean())/artist_info_df['popularity'].std()


# In[17]:


artist_info_df.isna().sum()


# In[18]:


genres_inv_map={}
counter=0
for g_lst in artist_info_df['genres']:
    for g in g_lst:
        if g not in genres_inv_map:
            genres_inv_map[g]= counter
            counter = counter +1
genres_inv_map


# In[19]:


len(genres_inv_map)


# In[20]:


artist_info_df= artist_info_df.set_index('artist_code')


# ## Define node features

# In[21]:


tracks_info['mode'].value_counts()


# In[22]:


tracks_info['key'].value_counts()


# In[23]:


tracks_info['speechiness'].value_counts()


# In[24]:


tracks_info['instrumentalness'].value_counts()


# In[25]:


tracks_info.corr()


# In[26]:


for c in 'energy loudness speechiness acousticness liveness instrumentalness valence danceability tempo'.split():
    tracks_info[f'{c}_norm']=(tracks_info[c]-tracks_info[c].mean())/tracks_info[c].std()


# In[27]:


tracks_info


# #### Artist features

# In[28]:


artist_features= artist_info_df['followers_norm popularity_norm'.split()].values
artist_features, artist_features.shape


# ## Generate daily graphs

# In[29]:


artists_map_inv= {v:i for i,v in artists_map.items()}
track_map_inv= {v:i for i,v in track_map.items()}
print(len(artists_map_inv))


# In[30]:


min_date= tracks_info['release_date'].min()
max_date= tracks_info['release_date'].max()

print(min_date, max_date)


# In[31]:


all_dates= [min_date+timedelta(days=x) for x in range((max_date-min_date).days)]
n_dates= len(all_dates)
print(n_dates)


# In[32]:


n_snapshots= n_dates // time_window_size


# In[33]:


def sample_negative_edges(positive_edges, nodes_to_code, n_samples):
    
    edges_= np.vstack((positive_edges[0],positive_edges[1])).T
    
    origins = []
    dest = []
    #Here we generate a negative link for each node in the snapshot
    for i in range(n_samples):
        
        o_node= np.random.randint(len(nodes_to_code.keys()))
        d_node= np.random.randint(len(nodes_to_code.keys()))

        edge_is_new= (len(edges_[(edges_[:,0]==o_node) & (edges_[:,1]==d_node)])==0)
        
        #If the (o_node, d_node) tuple already exists, try again...
        while not edge_is_new:
            d_node= np.random.randint(len(nodes_to_code.keys()))
        
            edge_is_new= (len(edges_[(edges_[:,0]==o_node) & (edges_[:,1]==d_node)])==0)

        origins.append(o_node)
        dest.append(d_node)
        
    edge_index_negs = torch.row_stack([torch.LongTensor(origins), torch.LongTensor(dest)])
    edge_label_negs= torch.zeros(edge_index_negs.shape[1]).reshape(-1,1).float()
    
    return edge_index_negs, edge_label_negs

def generate_artist_col_artist_edges(target_collab, artist_map_inv, edge_label_value):
    artist_col_artist=[]
    for i, song in tqdm(target_collab.iterrows(), desc='Generate artist_col_artist edges...', leave=False):
        artist_code_lst= song['artist_id']
        for a1,a2 in list(itertools.combinations(artist_code_lst, 2)):
            try:
                code_1= artists_map_inv[a1]
                code_2= artists_map_inv[a2]
                artist_col_artist.append([code_1,code_2])
            except:
                print(f"Error for artists {a1}, {a2}")

    artist_col_artist= np.array(artist_col_artist).T
    artist_col_artist_labels= torch.ones(artist_col_artist.shape[1], dtype=torch.float).reshape(-1,1)
    
    return torch.tensor(artist_col_artist), artist_col_artist_labels
    
    
def generate_daily_hetero_graph(tracks, 
                               artist_map_inv, 
                               songs_map_inv, 
                               target_dates_,
                               edge_type,
                               data=None):
    
    if data is None:
        data = HeteroData()

    training_tracks= tracks[tracks['release_date'].isin(target_dates_)]    
    training_collabs_df= training_tracks[training_tracks['num_artists']>1]
    
    edge_label=""
    if edge_type =='new':
        edge_label='new'
    elif edge_type == 'target':
        edge_label= 'target'
    
                
    #1. Generate edges
    # 2.1.1 artist_collaborate_artist 
    
    artist_col_artist, artist_col_artist_labels= generate_artist_col_artist_edges(training_collabs_df, artist_map_inv, edge_label)
    data['artist',f'{edge_label}collaborate','artist'].edge_index=artist_col_artist
    data['artist',f'{edge_label}collaborate','artist'].edge_label=artist_col_artist_labels
        
    
    
    if edge_type != 'target':
        #2.3 Artist_produces_track
        artist_produces_song=[]
        for i, song in tqdm(training_tracks.iterrows(), desc='Artist produces track...', leave=False):
            song_id= song['track_id']
            song_code= songs_map_inv[song_id]
            song_artists_lst= song['artist_id']
            #if len(song_artists_lst) > 1:
            for artist_id in song_artists_lst:
                try:
                    artist_code= artists_map_inv[artist_id]
                    artist_produces_song.append([artist_code,song_code])
                except:
                    print(f'Artist {artist_id} not included in the map')
        artist_produces_song= np.array(artist_produces_song).T
        artist_produces_song_labels= torch.ones(artist_produces_song.shape[1],dtype=torch.long).reshape(-1,1)

        #2.3 Artist has genres
        artist_has_genre=[]
        for artist_id, artist_info in tqdm(artist_info_df.iterrows(), desc='Artist has genres...', leave=False):
            artist_code= artists_map_inv[artist_id]
            artist_genres= artist_info['genres']
            for g in artist_genres:
                genre_code= genres_inv_map[g]
                artist_has_genre.append([artist_code, genre_code])
        artist_has_genre= np.array(artist_has_genre).T
        artist_has_genre_labels= torch.ones(artist_has_genre.shape[1],dtype=torch.long).reshape(-1,1)    

        #2.4 Track has genres
        song_has_genre=[]
        for i, song in tqdm(training_tracks.iterrows(), desc='Track has genres...', leave=False):
            song_id= song['track_id']
            song_code= songs_map_inv[song_id]
            song_genres_lst= song['genres']
            if len(song_genres_lst) > 1:
                for genre in song_genres_lst:
                    genre_code= genres_inv_map[genre]
                    song_has_genre.append([song_code,genre_code])
        song_has_genre= np.array(song_has_genre).T
        song_has_genre_labels= torch.ones(song_has_genre.shape[1],dtype=torch.long).reshape(-1,1)

       
        data['artist',f'{edge_label}produce','song'].edge_index=torch.tensor(artist_produces_song)
        data['artist',f'{edge_label}produce','song'].edge_label=artist_produces_song_labels

        data['artist',f'{edge_label}has','genre'].edge_index=torch.tensor(artist_has_genre)
        data['artist',f'{edge_label}has','genre'].edge_label=artist_has_genre_labels    

        data['song',f'{edge_label}has','genre'].edge_index=torch.tensor(song_has_genre)
        data['song',f'{edge_label}has','genre'].edge_label=song_has_genre_labels 
        
    return data


# In[34]:


def generate_dataset(n_snapshots,T, T_size):    
    
    datasets_path= os.path.join('data','generated', 'graphs_V4')
    isExist = os.path.exists(datasets_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(datasets_path)
        print("The dataset directory is created!")

    print("Generating new datasets...", end='')
    dataset= {}
    dataset['features']= {}
    
    tracks_features=tracks_info['energy_norm loudness_norm speechiness_norm acousticness_norm liveness_norm instrumentalness valence danceability tempo'.split()].values
    dataset['features']['tracks.num_nodes']= len(tracks_features)
    dataset['features']['tracks.x']=torch.tensor(tracks_features, dtype=torch.float)
    dataset['features']['tracks.node_id']=torch.arange(len(tracks_features))
    
    dataset['features']['artist.num_nodes'] = len(artist_features)
    dataset['features']['artist.x']= torch.tensor(artist_features, dtype=torch.float)
    dataset['features']['artist.node_id']= torch.arange(len(artist_features))
    
    dataset['features']['genre.num_nodes']= len(genres_inv_map)
    dataset['features']['genre.x'] = torch.ones(len(genres_inv_map)).reshape(-1,1).float()
    dataset['features']['genre.node_id']= torch.arange(len(genres_inv_map))

    dataset['edges']={}
    for i in tqdm(range(0,n_snapshots-1-T-T_size), desc='Time windows...'):
    #for i in tqdm(range(0,2), desc='Time windows...'):
        
        dates_new= all_dates[i *time_window_size: (i+1)* time_window_size]
        edges_new= generate_daily_hetero_graph(tracks_info,
                                                artists_map_inv, 
                                                track_map_inv, 
                                                dates_new,
                                                edge_type='new')
                
        dates_all= all_dates[:(i+1) * time_window_size]
        edges_all= generate_daily_hetero_graph(tracks_info,
                                                artists_map_inv, 
                                                track_map_inv, 
                                                dates_all,
                                                edge_type='all',
                                               data= edges_new)
        
        dates_target= all_dates[(i+1+T) * time_window_size:(i+1+T+T_size) * time_window_size]
        edges_all_w_targets= generate_daily_hetero_graph(tracks_info,
                                                         artists_map_inv, 
                                                         track_map_inv, 
                                                         dates_target,
                                                         edge_type='target',
                                                         data= edges_all)
        
        edges_all_w_targets_undirected = ToUndirected(merge=False)(edges_all_w_targets)

        dataset['edges'][i]= edges_all_w_targets_undirected
    
    graph_data_path=os.path.join('data', 'generated', 'graphs_V4', f'full_dataset_W_{time_window_size}_T_{T}_T_size_{T_size}.graph')
    torch.save(dataset, graph_data_path)

    print("DONE!")
    return dataset

T= 6
T_size=24
dataset=generate_dataset(n_snapshots, T, T_size)


# In[35]:


print(dataset)


# In[36]:


dataset['edges'][0]


# ----------------------

# In[37]:


"""
from torch_geometric.utils import degree
from collections import Counter

# Get list of degrees for each node
degrees = degree(data.edge_index[0]).numpy()

# Count the number of nodes for each degree
numbers = Counter(degrees)

# Bar plot
fig, ax = plt.subplots(figsize=(18, 7))
ax.set_xlabel('Node degree')
ax.set_ylabel('Number of nodes')
plt.bar(numbers.keys(),
        numbers.values(),
        color='#0A047A')
"""


# In[38]:


print("That's all folks!")

