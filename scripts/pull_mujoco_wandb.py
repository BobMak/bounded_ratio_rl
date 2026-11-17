"""Read-only W&B snapshot for the MuJoCo training-curve report."""
import argparse
import json
import base64
import netrc
import os
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def query(document, variables):
    key = os.environ.get('WANDB_API_KEY') or netrc.netrc().authenticators('api.wandb.ai')[2]
    request = urllib.request.Request('https://api.wandb.ai/graphql',
        data=json.dumps(dict(query=document, variables=variables)).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Basic ' + base64.b64encode(('api:' + key).encode()).decode()})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            if result.get('errors'):
                raise RuntimeError(result['errors'])
            return result['data']
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def runs(project):
    entity, name = project.split('/')
    cursor = None
    while True:
        result = query('''query($entity:String!, $project:String!, $cursor:String) {
          project(name:$project, entityName:$entity) {
            runs(first:100, after:$cursor) {
              pageInfo { endCursor hasNextPage }
              edges { node { name displayName group state createdAt config summaryMetrics } }
            }
          }
        }''', dict(entity=entity, project=name, cursor=cursor))['project']['runs']
        yield from (edge['node'] for edge in result['edges'])
        print('Fetched run page', len(result['edges']), flush=True)
        if not result['pageInfo']['hasNextPage']:
            break
        cursor = result['pageInfo']['endCursor']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='vladmak/BPO')
    parser.add_argument('--output', type=Path, default=Path('results/mujoco_training_curves'))
    parser.add_argument('--histories', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.histories:
        records = []
        for run in runs(args.project):
            cfg = {k: v.get('value') for k, v in json.loads(run['config']).items()}
            env = cfg.get('env', {})
            if not isinstance(env, dict) or not any(x in env.get('env_id', '') for x in
                    ['Ant-', 'HalfCheetah-', 'Hopper-', 'Humanoid-', 'HumanoidStandup-',
                     'Swimmer-', 'Walker2d-', 'Reacher-', 'Pusher-', 'InvertedPendulum-',
                     'InvertedDoublePendulum-']):
                continue
            records.append(dict(id=run['name'], name=run['displayName'], group=run['group'], state=run['state'],
                                created_at=run['createdAt'], url='https://wandb.ai/' + args.project + '/runs/' + run['name'],
                                config={k: cfg[k] for k in ['env', 'algo', 'training'] if k in cfg},
                                summary=json.loads(run['summaryMetrics'] or '{}')))
        data = dict(project=args.project, fetched_at=datetime.now(timezone.utc).isoformat(), runs=records)
        (args.output / 'inventory.json').write_text(json.dumps(data, indent=2))
        print('MuJoCo runs:', len(records), flush=True)
        for (env, group), n in sorted(Counter((r['config']['env']['env_id'], r['group'] or '') for r in records).items()):
            print(env, group or '(ungrouped)', n)
    else:
        from concurrent.futures import ThreadPoolExecutor
        data = json.loads((args.output / 'inventory.json').read_text())
        history_dir = args.output / 'histories'
        history_dir.mkdir(exist_ok=True)
        records = [r for r in data['runs'] if 'rollout/ep_rew_mean' in r['summary']]
        def pull(record):
            path = history_dir / (record['id'] + '.json')
            if path.exists():
                return
            # Page by W&B history step; request at least one sample per possible row.
            # This is the GraphQL equivalent of the SDK's legacy complete-history scan.
            entity, project = args.project.split('/')
            rows = []
            for start in range(0, int(record['summary'].get('_step', 0)) + 1, 1000):
                spec = dict(keys=['_step', 'global_step', 'rollout/ep_rew_mean'],
                            xAxis='_step', minStep=start, maxStep=start + 1000, samples=1000)
                result = query('''query($entity:String!, $project:String!, $run:String!, $specs:[JSONString!]!) {
                  project(name:$project, entityName:$entity) {
                    run(name:$run) { sampledHistory(specs:$specs) }
                  }
                }''', dict(entity=entity, project=project, run=record['id'], specs=[json.dumps(spec)]))
                rows.extend(result['project']['run']['sampledHistory'][0])
            path.write_text(json.dumps(rows))
            print(record['id'], record['name'], len(rows), flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(pull, records))


if __name__ == '__main__':
    main()
