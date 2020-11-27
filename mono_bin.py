import numpy as np
import pandas as pd
import warnings

'''
Data monotonic binning utils for credit score card.
'''

_COLS = pd.Index([
    'bin_no',
    'bin',
    'total',
    'total_pct',
    'bad',
    'bad_pct',
    'good',
    'good_pct',
    'bad_rate',
    'woe',
    'woe_diff',
    'iv',
])


def convert_discrete_to_continuous(df: pd.DataFrame, x='x', y='y', return_map=False):
    data = df[[x, y]].rename(columns={x: 'x', y: 'y'})
    mapping = data.groupby('x')['y'].agg([('new_x', 'mean')]).reset_index(drop=False)
    new_data = data.merge(mapping, on='x', how='left')['new_x']
    new_data.index = data.index
    if return_map:
        return new_data, mapping
    else:
        return new_data


def mono_bin_fit(df: pd.DataFrame,
                 x: str = 'x',
                 y: str = 'y',
                 max_bin_cnt: int = 5,
                 min_total: int = 5,
                 min_bad: int = 1,
                 min_good: int = 1,
                 min_woe_diff: float = 0.1,
                 right: bool = True,
                 ascending: bool = False,
                 try_na_separate: bool = False,
                 return_rule: bool = False
                 ):
    # print('step 1: input args check')

    assert isinstance(df, pd.DataFrame)
    assert x != y
    assert df[x].dtypes.kind in 'ifO' and df[x].notna().any()
    assert df[y].min() == 0 and df[y].max() == 1 and len(np.unique(df[y])) == 2
    assert max_bin_cnt >= 0
    assert min_bad >= 0
    assert min_good >= 0
    assert min_total > 0
    assert min_woe_diff > 0

    data = df[[x, y]]
    mask = data[x].isna()

    mp = None
    if data[x].dtypes.kind == 'O':
        if ascending:
            warnings.warn('Ascending args is reset to be False when binning discrete data automatically')
            ascending = False # reset automatically
        data = data.copy()
        new_x, mp = convert_discrete_to_continuous(data, x, y, return_map=True)
        data[x] = new_x

    na_total = data[mask][y].count()
    na_bad = data[mask][y].sum()
    na_good = na_total - na_bad

    notna_total = data[~mask][y].count()  # must > 0
    notna_bad = data[~mask][y].sum()
    notna_good = notna_total - notna_bad

    # print('step 2: not nan data mono merge')

    a = data[~mask].rename(columns={x: 'x', y: 'y'}).sort_values(by='x', ascending=ascending). \
        groupby('x', sort=False)['y'].agg([('total', 'count'), ('bad', 'sum'), ('bad_rate', 'mean')]).reset_index(
        drop=False)
    a['bad_rate'] = a['bad_rate'].astype('float')
    a['good'] = a['total'] - a['bad']
    a['del'] = 0
    a['woe_diff'] = np.nan
    a['begin'] = np.nan
    a['end'] = np.nan

    while True:
        merged = False
        idx = a[a['del'] == 0].index

        for i in range(len(idx)):
            if i == len(idx) - 1:
                break
            left_pos = idx[i]
            right_pos = idx[i + 1]
            if a.at[left_pos, 'bad_rate'] <= a.at[right_pos, 'bad_rate']:  # right merge to left
                a.at[left_pos, 'bad'] += a.at[right_pos, 'bad']
                a.at[left_pos, 'good'] += a.at[right_pos, 'good']
                a.at[left_pos, 'total'] += a.at[right_pos, 'total']
                a.at[left_pos, 'bad_rate'] = a.at[left_pos, 'bad'] / a.at[left_pos, 'total']
                a.at[right_pos, 'del'] = 1
                merged = True
                break

        if not merged:
            break

    assert len(a[a['del'] == 0]) >= 1
    assert notna_total > 0

    # print('step 3: nan data merge')

    if na_total:
        na_bad_rate = na_bad * 1.0 / na_total
        if try_na_separate and (pd.Series([na_total, na_bad, na_good, notna_total, notna_bad, notna_good]) >=
                                pd.Series([min_total, min_bad, min_good] * 2)).all():
            na_replace_val = np.nan  # nan separate
        else:
            na_near_idx = (a[a['del'] == 0]['bad_rate'] - na_bad_rate).abs().idxmin()
            na_replace_val = a.at[na_near_idx, 'x']  # nan merge into the bin which bad_rate nearest
            a.at[na_near_idx, 'bad'] += na_bad
            a.at[na_near_idx, 'good'] += na_good
            a.at[na_near_idx, 'total'] += na_total
            a.at[na_near_idx, 'bad_rate'] = a.at[na_near_idx, 'bad'] / a.at[na_near_idx, 'total']
    else:
        na_bad_rate = np.nan
        max_bad_rate_idx = (a[a['del'] == 0]['bad_rate']).idxmax()
        na_replace_val = a.at[max_bad_rate_idx, 'x']  # nan merge into the bin which bad_rate max

    # print('step 4: merge neighboring bins')

    big_number = 2 ** 31

    while True:
        merged = False
        idx = a[a['del'] == 0].index
        if len(idx) < 2:
            break

        for i in range(len(idx)):
            if i == len(idx) - 1:
                a.at[idx[i], 'woe_diff'] = np.nan  # reset
                break
            left_pos = idx[i]
            right_pos = idx[i + 1]
            # calc woe diff
            if a.at[left_pos, 'bad_rate'] >= 1 or a.at[right_pos, 'bad_rate'] <= 0:
                a.at[left_pos, 'woe_diff'] = min_woe_diff + big_number  # woe_diff very large
            else:
                a.at[left_pos, 'woe_diff'] = np.log(
                    (a.at[left_pos, 'bad_rate'] / (1 - a.at[left_pos, 'bad_rate']))
                    /
                    (a.at[right_pos, 'bad_rate'] / (1 - a.at[right_pos, 'bad_rate']))
                )  # normal woe diff
            if (a.loc[[left_pos, right_pos], ['total', 'bad', 'good']] < [min_total, min_bad, min_good]).any(axis=None):
                a.at[left_pos, 'woe_diff'] += -min_woe_diff - 2 * big_number  # woe_diff -inf

        pos = a[a['del'] == 0]['woe_diff'].reset_index(drop=True).idxmin()
        assert pos >= 0
        left_pos = idx[pos]
        right_pos = idx[pos + 1]

        if a.at[left_pos, 'woe_diff'] < min_woe_diff or len(a[a['del'] == 0]) > max_bin_cnt > 0:
            a.at[left_pos, 'bad'] += a.at[right_pos, 'bad']
            a.at[left_pos, 'good'] += a.at[right_pos, 'good']
            a.at[left_pos, 'total'] += a.at[right_pos, 'total']
            a.at[left_pos, 'bad_rate'] = a.at[left_pos, 'bad'] / a.at[left_pos, 'total']  # merged
            a.at[right_pos, 'del'] = 1  # merged, delete
            a.at[left_pos, 'woe_diff'] = np.nan  # reset
            a.at[right_pos, 'woe_diff'] = np.nan  # reset
            merged = True

        if not merged:
            break

    # print('step 5: calc bins, woe and rule')

    idx = a[a['del'] == 0].index
    for i in range(len(idx)):
        left_pos = idx[i]
        begin = a.at[left_pos, 'x']
        if i == len(idx) - 1:
            end = a.at[a.index[-1], 'x']
        else:
            right_pos = idx[i + 1]
            end = a.at[right_pos - 1, 'x']
        a.at[left_pos, 'begin'] = min(begin, end)
        a.at[left_pos, 'end'] = max(begin, end)

    b = a[a['del'] == 0].copy()
    b.sort_values(by='begin', axis=0, ascending=True, inplace=True, na_position='last')

    if mp is None:
        if right:
            bins = sorted(b['end'].shift(1).fillna(-np.inf)) + [np.inf]
        else:
            bins = sorted(b['begin'].shift(-1).fillna(-np.inf)) + [np.inf]
    else:
        bins = []

    no = 1
    for i, row in b.iterrows():
        b.at[i, 'bin_no'] = no
        min_val = row['begin']
        max_val = row['end']

        if mp is None:  # continuous
            if right:
                b.at[i, 'bin'] = '(%s, %s]' % (bins[no - 1], bins[no])
            else:
                b.at[i, 'bin'] = '[%s, %s)' % (bins[no - 1], bins[no])
        else:  # discrete
            vals = mp[(mp['new_x'] >= min_val) & (mp['new_x'] <= max_val)]['x'].tolist()
            bins.append(vals)
            assert len(bins) == no
            b.at[i, 'bin'] = '{%s}' % ','.join(str(e) for e in bins[no - 1])

        if min_val <= na_replace_val <= max_val:
            if mp is None:  # continuous
                b.at[i, 'bin'] += ' or nan'
            else:  # discrete
                bins[no - 1].append(np.nan)
                b.at[i, 'bin'] = '{%s}' % ','.join(str(e) for e in bins[no - 1])

        no += 1

    if np.isnan(na_replace_val):
        b.loc[-1, ['total', 'bad', 'good', 'bad_rate', 'del', 'bin', 'bin_no']] = [
            na_total, na_bad, na_good, na_bad_rate, 0, 'nan' if mp is None else '{nan}', no]
        if mp is not None:
            bins.append([np.nan])

    # print('step 6: return')

    c = b.drop(columns=['x', 'begin', 'end', 'del']).reset_index(drop=True)

    c['total_pct'] = c['total'] / c['total'].sum()
    c['bad_pct'] = c['bad'] / c['bad'].sum()
    c['good_pct'] = c['good'] / c['good'].sum()
    c['woe'] = np.log(c['bad_pct'] / c['good_pct'])
    c['iv'] = (c['bad_pct'] - c['good_pct']) * c['woe']
    c['woe_diff'] = c['woe'] - c['woe'].shift(-1)

    if return_rule:
        rule = {
            'bins': bins,
            'woe': c['woe'].tolist(),
            'iv': c['iv'].sum(),
            'raw_is_continuous': mp is None
        }

        if mp is None:
            rule['right'] = right
            if not np.isnan(na_replace_val):
                rule['na_replace_val'] = na_replace_val

        return c[_COLS], rule
    else:
        return c[_COLS]


def mono_bin_transform(x: pd.Series, rule: dict):
    raw_is_continuous = rule['raw_is_continuous']
    bins = rule['bins']
    woe = rule['woe']

    data = pd.DataFrame({'x': pd.Series(x)})

    if raw_is_continuous:
        assert data['x'].dtypes.kind in 'if'
        na_replace_val = rule.get('na_replace_val')
        if not pd.api.types.is_number(na_replace_val):
            na_replace_val = np.nan
        right = bool(rule.get('right'))
        data['bin_no'] = np.digitize(data['x'].fillna(na_replace_val), bins=bins, right=right)
        data['bin'] = pd.cut(data['x'].fillna(na_replace_val), bins=bins, right=right).astype('str')
        na_bin_no = np.digitize(na_replace_val, bins=bins, right=right)
        if na_bin_no < len(bins):
            data.loc[data['bin_no'] == na_bin_no, 'bin'] += ' or nan'
        new_data = data.merge(pd.DataFrame([{'bin_no': i + 1, 'woe': w} for i, w in enumerate(woe)]),
                              how='left', on='bin_no')
        new_data.index = data.index
    else:
        assert data['x'].dtypes.kind in 'O'
        assert len(bins) == len(woe)
        mp = []
        for idx, xs in enumerate(bins):
            b = '{%s}' % ','.join(str(s) for s in xs)
            for s in xs:
                mp.append([s, idx + 1, b, woe[idx]])
        mp = pd.DataFrame(mp, columns=['x', 'bin_no', 'bin', 'woe'])
        assert mp['x'].isna().any()
        mask = data['x'].isin(mp['x'])
        data.loc[~mask, 'x'] = np.nan
        new_data = data.merge(mp, how='left', on='x')
        new_data.index = data.index
        new_data['x'] = pd.Series(x)
    assert len(new_data) == len(x)
    return new_data


def mono_bin_adjust(df: pd.DataFrame, rule: dict, x: str = 'x', y: str = 'y'):
    assert isinstance(df, pd.DataFrame)
    assert x != y
    assert df[x].dtypes.kind in 'ifO' and df[x].notna().any()
    assert df[y].min() == 0 and df[y].max() == 1 and len(np.unique(df[y])) == 2

    raw_is_continuous = bool(rule['raw_is_continuous'])
    bins = rule['bins']

    data = df[[x, y]].rename(columns={x: 'x', y: 'y'})

    rule_new = dict()
    rule_new['raw_is_continuous'] = raw_is_continuous
    rule_new['bins'] = bins

    if raw_is_continuous:
        assert data['x'].dtypes.kind in 'if'
        assert -np.inf in bins
        assert np.inf in bins
        na_replace_val = rule.get('na_replace_val')
        if not pd.api.types.is_number(na_replace_val):
            na_replace_val = np.nan
        right = bool(rule.get('right'))
        rule_new['right'] = right
        data['bin'] = pd.cut(data['x'].fillna(na_replace_val), bins=bins, right=right).astype('str')
        data['bin_no'] = np.digitize(data['x'].fillna(na_replace_val), bins=bins, right=right)
        na_bin_no = np.digitize(na_replace_val, bins=bins, right=right)
        if na_bin_no < len(bins):
            assert len(np.unique(data['bin_no'])) == len(bins) - 1  # # since no data in some bin
            data.loc[data['bin_no'] == na_bin_no, 'bin'] += ' or nan'
        else:
            pass
            assert len(np.unique(data['bin_no'])) == len(bins)  # since no data in some bin
        if not np.isnan(na_replace_val):
            rule_new['na_replace_val'] = na_replace_val
    else:
        assert data['x'].dtypes.kind in 'O'
        mp = []
        for idx, xs in enumerate(bins):
            assert isinstance(xs, (set, list))
            bin = '{%s}' % ','.join(str(s) for s in xs)
            for s in xs:
                mp.append([s, bin, idx + 1])
        mp = pd.DataFrame(mp, columns=['x', 'bin', 'bin_no', ])
        assert mp['x'].dtypes.kind in 'O'
        assert mp['x'].isna().any()
        mask = data['x'].isin(mp['x'])
        data.loc[~mask, 'x'] = np.nan
        data = data.merge(mp, how='left', on='x')
        assert len(data) == len(df[x])
        data['x'] = df[x]

    c = data.groupby(['bin_no', 'bin'])['y'].agg(
        [('total', 'count'), ('bad', 'sum'), ('bad_rate', 'mean')]).reset_index(
        drop=False)
    c['bad_rate'] = c['bad_rate'].astype('float')
    c['good'] = c['total'] - c['bad']
    c['total_pct'] = c['total'] / c['total'].sum()
    c['bad_pct'] = c['bad'] / c['bad'].sum()
    c['good_pct'] = c['good'] / c['good'].sum()
    c['woe'] = np.log(c['bad_pct'] / c['good_pct'])
    c['iv'] = (c['bad_pct'] - c['good_pct']) * c['woe']
    c['woe_diff'] = c['woe'] - c['woe'].shift(-1)

    rule_new['woe'] = c['woe'].tolist()
    rule_new['iv'] = c['iv'].sum()

    return c[_COLS], rule_new
