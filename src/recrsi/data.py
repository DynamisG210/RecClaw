"""Chronological sequential task construction from a user/item/timestamp TSV."""
import csv
from collections import defaultdict
from pathlib import Path
from .config import sha
from .prepare import write


def split(interactions, output, test_output, dataset='cds', max_length=50):
    interactions=Path(interactions).resolve(strict=True)
    output,test_output=Path(output).resolve(),Path(test_output).resolve()
    if max_length<1 or not dataset or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in dataset):
        raise ValueError('Use a positive max length and a simple dataset name')
    if test_output.is_relative_to(output):
        raise ValueError('Store held-out TEST outside the research data directory')
    if output.exists() or test_output.exists():
        raise FileExistsError('Use fresh split output paths')
    histories=defaultdict(list)
    with interactions.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if reader.fieldnames != ['user_id','item_id','timestamp']:
            raise ValueError('Expected TSV columns: user_id, item_id, timestamp')
        for index,row in enumerate(reader):
            user,item=row['user_id'],row['item_id']
            if not user or not item or any(c.isspace() for c in user+item):
                raise ValueError('IDs must be nonempty tokens without whitespace')
            histories[user].append((int(row['timestamp']),index,item))
    sequences={}
    duplicates=0
    for user,events in histories.items():
        seen=set();items=[]
        for _,_,item in sorted(events):
            if item in seen:duplicates+=1;continue
            seen.add(item);items.append(item)
        if len(items)>=4:sequences[user]=items
    if not sequences:
        raise ValueError('Need users with at least four distinct chronological items')
    output.mkdir(parents=True)
    test_output.parent.mkdir(parents=True,exist_ok=True)
    header='user_id:token\titem_id_list:token_seq\titem_id:token\n'
    def row(user,prefix,target):return user+'\t'+' '.join(prefix[-max_length:])+'\t'+target+'\n'
    with (output/(dataset+'.train.inter')).open('w',encoding='utf-8') as train, \
         (output/(dataset+'.dev.inter')).open('w',encoding='utf-8') as dev, \
         test_output.open('x',encoding='utf-8') as test:
        train.write(header);dev.write(header);test.write(header)
        for user,items in sequences.items():
            for i in range(1,len(items)-2):train.write(row(user,items[:i],items[i]))
            dev.write(row(user,items[:-2],items[-2]))
            test.write(row(user,items[:-1],items[-1]))
    (output/(dataset+'.user')).write_text('user_id:token\n'+'\n'.join(sequences)+'\n',encoding='utf-8')
    catalog=sorted({item for items in sequences.values() for item in items})
    (output/(dataset+'.item')).write_text('item_id:token\n'+'\n'.join(catalog)+'\n',encoding='utf-8')
    report=dict(protocol='chronological-first-occurrence-leave-two-out-v1',input_sha256=sha(interactions),
                users=len(sequences),items=len(catalog),users_below_four_items=len(histories)-len(sequences),
                removed_repeated_interactions=duplicates,max_length=max_length,
                test_output=str(test_output),test_sha256=sha(test_output))
    write(output/'SPLIT.json',report)
    return report
