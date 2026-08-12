#!/usr/bin/env python3
from pathlib import Path
from mizan.data import chronological_split, load_arafacts, stratified_split

frame = load_arafacts(Path('/home/ubuntu/arafacts_source/Dataset/AraFacts/AraFacts.csv'), Path('/home/ubuntu/arafacts_source/Dataset/AraFacts/AraFacts_content.csv'))
chron = chronological_split(frame)
print('chronological')
for name, part in [('train', chron.train), ('validation', chron.validation), ('test', chron.test)]:
    print(name, len(part), part['normalized_label'].value_counts().to_dict())
print('stratified')
train, test = stratified_split(frame, test_size=0.2)
print('train', len(train), train['normalized_label'].value_counts().to_dict())
print('test', len(test), test['normalized_label'].value_counts().to_dict())
print('date ranges')
for name, part in [('train', chron.train), ('validation', chron.validation), ('test', chron.test)]:
    print(name, part['date_parsed'].min(), part['date_parsed'].max())
