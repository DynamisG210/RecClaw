import csv
import pytest
from recrsi.data import split


def test_split_orders_deduplicates_and_separates_test(tmp_path):
    raw=tmp_path/'raw.tsv'
    raw.write_text('user_id\titem_id\ttimestamp\nu\ti4\t4\nu\ti1\t1\nu\ti2\t2\nu\ti1\t5\nu\ti3\t3\nshort\ti1\t1\n')
    report=split(raw,tmp_path/'search',tmp_path/'heldout/test.inter',max_length=2)
    assert report['removed_repeated_interactions']==1
    assert report['users_below_four_items']==1
    assert (tmp_path/'search/cds.train.inter').read_text().splitlines()[1]=='u\ti1\ti2'
    assert (tmp_path/'search/cds.dev.inter').read_text().splitlines()[1]=='u\ti1 i2\ti3'
    assert (tmp_path/'heldout/test.inter').read_text().splitlines()[1]=='u\ti2 i3\ti4'
    assert not list((tmp_path/'search').glob('*.test.inter'))
    with pytest.raises(FileExistsError):split(raw,tmp_path/'search',tmp_path/'heldout/test.inter')
