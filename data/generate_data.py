import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate dates for the last year
base_date = datetime.now() - timedelta(days=365)
dates = [base_date + timedelta(days=random.randint(0, 365)) for _ in range(1000)]

# Communities and sub-communities
communities = ['Dubai Marina', 'Downtown Dubai', 'Business Bay', 'Jumeirah Lakes Towers', 
               'Palm Jumeirah', 'Dubai Hills Estate', 'Al Barsha', 'Jumeirah Village Circle']

sub_communities = {
    'Dubai Marina': ['Marina Promenade', 'Dubai Marina Mall', 'Marina Walk'],
    'Downtown Dubai': ['Burj Khalifa Area', 'Dubai Mall', 'Opera District'],
    'Business Bay': ['Bay Avenue', 'Executive Towers', 'Marasi Drive'],
    'Jumeirah Lakes Towers': ['Cluster A', 'Cluster B', 'Cluster C'],
    'Palm Jumeirah': ['Golden Mile', 'Shoreline', 'Crescent'],
    'Dubai Hills Estate': ['Park Heights', 'Collective', 'Golf Place'],
    'Al Barsha': ['Barsha Heights', 'Mall of the Emirates', 'Al Barsha 1'],
    'Jumeirah Village Circle': ['District 12', 'District 13', 'District 14']
}

# Property types and views
property_types = ['Apartment', 'Villa', 'Townhouse']
views = ['Sea', 'Marina', 'Skyline', 'Park', 'Community']

# Generate data
data = []
for _ in range(1000):
    community = random.choice(communities)
    sub_community = random.choice(sub_communities[community])
    
    data.append({
        'rent_aed': round(random.uniform(2000, 200000), 2),
        'bedrooms': random.randint(1, 5),
        'bathrooms': random.randint(1, 6),
        'size_sqft': random.randint(400, 10000),
        'floor': random.randint(1, 80),
        'year_built': random.randint(2000, 2025),
        'distance_to_metro_km': round(random.uniform(0.5, 10.0), 1),
        'distance_to_mall_km': round(random.uniform(1.0, 15.0), 1),
        'community': community,
        'sub_community': sub_community,
        'furnishing': random.choice(['Unfurnished', 'Furnished', 'Partly Furnished']),
        'property_type': random.choice(property_types),
        'view': random.choice(views),
        'has_pool': random.choice([0, 1]),
        'has_gym': random.choice([0, 1]),
        'has_balcony': random.choice([0, 1]),
        'chiller_free': random.choice([0, 1]),
        'bills_included': random.choice([0, 1]),
        'listing_date': dates.pop()
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df['listing_month'] = pd.to_datetime(df['listing_date']).dt.month
df['listing_quarter'] = pd.to_datetime(df['listing_date']).dt.quarter

# Save to CSV
df.to_csv('data/raw_listings.csv', index=False)