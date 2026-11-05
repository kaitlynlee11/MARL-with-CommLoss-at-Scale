"""Alternating constrained PPO for the original binary communication channel.

An empirical adaptation of the supplied ACR-PPO/CS-MARL papers, not a claim of
reproducing their algorithms or inheriting their theoretical guarantees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .collision_vector_env import CollisionVectorSpread
from .mappo import gae, load_policy
from .sequential import ModularMAPPO, constrained_advantage, critic_state, proximity_cost, select_movement, selection_score
from .train import adaptive_reward, validate
from .rewards import TASK_REWARD_VERSION


def clipped_loss(ratio,advantage,clip):
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()


def clipped_value_loss(prediction,old,target,clip=.2):
    limited=old+(prediction-old).clamp(-clip,clip)
    return .5*torch.maximum((prediction-target).square(),(limited-target).square()).mean()


def categorical_kl(old_log,new_log):
    return (old_log.exp()*(old_log-new_log)).sum(dim=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3200000,help='Additional world steps, independent of initializer lineage')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--envs',type=int,default=64)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--sequence-length',type=int,default=25)
    parser.add_argument('--actor-lr',type=float,default=.0001)
    parser.add_argument('--critic-lr',type=float,default=.0003)
    parser.add_argument('--clip',type=float,default=.1)
    parser.add_argument('--target-kl',type=float,default=.0075)
    parser.add_argument('--risk-budget',type=float,default=.02)
    parser.add_argument('--packet-budget',type=float,default=.18)
    parser.add_argument('--proximity-weight',type=float,default=2.)
    parser.add_argument('--eval-every',type=int,default=25)
    parser.add_argument('--validation-episodes',type=int,default=200)
    args=parser.parse_args()
    if (min(args.steps,args.threads,args.envs,args.epochs,args.eval_every,args.validation_episodes)<1
            or args.sequence_length<1 or 100%args.sequence_length or args.steps%(100*args.envs)
            or not 0<args.clip<1 or not 0<args.risk_budget<1 or not 0<args.packet_budget<.2
            or not all(np.isfinite(v) and v>0 for v in [args.actor_lr,args.critic_lr,args.target_kl])
            or not np.isfinite(args.proximity_weight) or args.proximity_weight<0):
        parser.error('invalid budgets, optimization parameters or sequence length')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    parent,checkpoint=load_policy(args.source)
    parent_config=checkpoint['config']
    assert parent_config['min_agent_distance']==.4 and parent_config['min_landmark_distance']==.4
    assert parent_config['loss']==0 and parent_config['terminate_on_collision']
    model=ModularMAPPO.from_joint(parent)
    args.output.mkdir(parents=True,exist_ok=False)
    initializer=args.source.read_bytes()
    (args.output/'initializer.pt').write_bytes(initializer)
    config={**parent_config,**vars(args),'source':str(args.source.resolve()),'output':str(args.output.resolve()),
            'algorithm':'modular constrained PPO; paper-inspired binary-gate adaptation',
            'task_reward_version':TASK_REWARD_VERSION,
            'reward_formula':'coverage_bonus*covered_count + success_bonus*success - collision_penalty*pairs - packet_penalty*attempted_links - time_penalty*(not success) - distance_penalty*distance_sum*(not full_coverage) - failure_penalty*collision_terminal - proximity_weight*proximity_cost',
            'parent_sha256':hashlib.sha256(initializer).hexdigest(),'parent_steps':checkpoint['steps'],
            'training_schedule':'adaptive','model_spec':model.spec,'selection_rule':'traffic<20%, then success - 2*collision fraction',
            'actor_observations':28,'critic_clock_only':True,'validation_seed':100000,
            'reward_bootstrap':'zero at collision and scheduled finite-horizon end',
            'cost_advantages':'undiscounted episode packet ratio and collision indicator; separate bounded critics',
            'proximity_margin':.45,'packet_dual_rate':10.,'risk_dual_rate':20.,'dual_upper_bound':1000.,
            'initial_packet_multiplier':50.,'initial_risk_multiplier':100.,
            'comparison_test_seeds':'400000-400199; fresh 600000-600199 after checkpoint selection'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    schedule_optimizer=torch.optim.Adam(model.actor.scheduler.parameters(),lr=args.actor_lr,eps=1e-5)
    behavior_optimizer=torch.optim.Adam(model.actor.behavior.parameters(),lr=args.actor_lr,eps=1e-5)
    critic_parameters=list(model.critic.parameters())+list(model.packet_critic.parameters())+list(model.risk_critic.parameters())
    value_optimizer=torch.optim.Adam(critic_parameters,lr=args.critic_lr,eps=1e-5)
    env=CollisionVectorSpread(batch=args.envs,seed=args.seed+checkpoint['steps'],horizon=100,loss=0,min_agent_distance=.4,min_landmark_distance=.4)
    env.reset_indices(np.ones(args.envs,dtype=bool))
    obs,state=env.observe(),critic_state(env)
    memory=torch.zeros(args.envs,3,2,model.spec['hidden'])
    episode_packets=np.zeros(args.envs)
    packet_multiplier,risk_multiplier=50.,100.
    best=None
    total=0
    update=0
    started=time.perf_counter()
    validation_args=dict(episodes=args.validation_episodes,seed=100000,horizon=100,radius=.6,
                         schedule='adaptive',loss=0,min_landmark_distance=.4,min_agent_distance=.4,terminate_on_collision=True)

    def save(name,validation):
        torch.save({'model':model.state_dict(),'model_spec':model.spec,'steps':total,
                    'config':config,'validation':validation,'packet_multiplier':packet_multiplier,'risk_multiplier':risk_multiplier,
                    'scheduler_optimizer':schedule_optimizer.state_dict(),'behavior_optimizer':behavior_optimizer.state_dict(),
                    'critic_optimizer':value_optimizer.state_dict()},args.output/name)

    def status(state,validation):
        (args.output/'status.json').write_text(json.dumps({'state':state,'steps':total,'step_limit':args.steps,
            'validation':validation,'packet_multiplier':packet_multiplier,'risk_multiplier':risk_multiplier},indent=2),encoding='utf-8')

    initial=validate(model,**validation_args)
    (args.output/'initial-validation.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    best=selection_score(initial)
    save('best.pt',initial)
    save('initial.pt',initial)
    status('running',initial)
    validation=initial
    with (args.output/'learning.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while total<args.steps:
            update+=1
            buffer={key:[] for key in ['obs','state','memory','gate','move','glog','blog','gall','ball','rv','pv','sv',
                                       'reward','packet','risk','nr','np','ns','done']}
            finished_packets,finished_risks=[],[]
            for _ in range(100):
                with torch.no_grad():
                    gd,conditional,next_memory=model.actor.components(torch.from_numpy(obs),memory)
                    gates=gd.sample()
                    movement=select_movement(conditional,gates)
                    moves=movement.sample()
                    rv=model.value(torch.from_numpy(state)).numpy()
                    pv,sv=[v.numpy() for v in model.costs(torch.from_numpy(state))]
                next_obs,_,_,done,info=env.step(moves.numpy(),gates.numpy())
                next_state=critic_state(env)
                packet=(info['attempted_links']/600).astype(np.float32)
                risk=info['collision_terminal'].astype(np.float32)
                # Base task objective is retained; soft proximity cost is a training-only
                # adaptation of the pre-collision penalty in ACR-PPO Appendix D.1.
                reward=adaptive_reward(info,argparse.Namespace(**parent_config),0)
                reward-=parent_config['failure_penalty']*risk
                reward-=args.proximity_weight*proximity_cost(env.pos)
                with torch.no_grad():
                    nr=model.value(torch.from_numpy(next_state)).numpy()
                    npv,nsv=[v.numpy() for v in model.costs(torch.from_numpy(next_state))]
                values={'obs':obs,'state':state,'memory':memory.numpy().copy(),'gate':gates.numpy(),'move':moves.numpy(),
                        'glog':gd.log_prob(gates).numpy(),'blog':movement.log_prob(moves).numpy(),
                        'gall':gd.logits.numpy(),'ball':torch.log_softmax(conditional,dim=-1).numpy(),
                        'rv':rv,'pv':pv,'sv':sv,'reward':reward,'packet':packet,'risk':risk,'nr':nr,'np':npv,'ns':nsv,'done':done}
                for key,value in values.items():
                    buffer[key].append(value)
                episode_packets+=packet
                finished_packets.extend(episode_packets[done].tolist())
                finished_risks.extend(risk[done].tolist())
                episode_packets[done]=0
                if done.any():
                    env.reset_indices(done)
                    obs,state=env.observe(),critic_state(env)
                else:
                    obs,state=next_obs,next_state
                memory=next_memory*torch.from_numpy(~done)[:,None,None,None]
                total+=args.envs
            a={key:np.asarray(value) for key,value in buffer.items()}
            ra,returns=gae(a['reward'],a['rv'],a['nr'],a['done'],terminals=a['done'],gamma=.99)
            pa,packet_returns=gae(a['packet'],a['pv'],a['np'],a['done'],terminals=a['done'],gamma=1.)
            sa,risk_returns=gae(a['risk'],a['sv'],a['ns'],a['done'],terminals=a['done'],gamma=1.)
            advantages=constrained_advantage(ra,pa,sa,packet_multiplier,risk_multiplier)
            count=100*args.envs
            observations=torch.from_numpy(a['obs'].reshape(count,3,28))
            states=torch.from_numpy(a['state'].reshape(count,37))
            gates=torch.from_numpy(a['gate'].reshape(count,3))
            moves=torch.from_numpy(a['move'].reshape(count,3))
            old_glog=torch.from_numpy(a['glog'].reshape(count,3))
            old_blog=torch.from_numpy(a['blog'].reshape(count,3))
            old_gall=torch.from_numpy(a['gall'].reshape(count,3,2))
            old_ball=torch.from_numpy(a['ball'].reshape(count,3,2,5))
            advantage=torch.from_numpy(advantages.reshape(count,1))
            reward_target=model.normalize_value_targets(torch.from_numpy(returns.reshape(count)))
            old_value=(torch.from_numpy(a['rv'].reshape(count))-model.value_mean)/model.value_std
            ptarget=torch.from_numpy(packet_returns.reshape(count))
            starget=torch.from_numpy(risk_returns.reshape(count))
            old_p=torch.from_numpy(a['pv'].reshape(count))
            old_s=torch.from_numpy(a['sv'].reshape(count))
            sequences=count//args.sequence_length
            order=rng.permutation(sequences)
            initial_t=(order//args.envs)*args.sequence_length
            worlds=order%args.envs
            indices=(initial_t[None]+np.arange(args.sequence_length)[:,None])*args.envs+worlds[None]
            idx=indices.reshape(-1)

            def replay(network,branch):
                hidden=torch.from_numpy(a['memory'][initial_t,worlds,:,branch,:])
                logits=[]
                for ti in indices:
                    value,hidden=network(observations[ti],hidden)
                    logits.append(value)
                    reset=torch.from_numpy(a['done'].reshape(count)[ti])
                    hidden=hidden*(~reset)[:,None,None]
                return torch.stack(logits).reshape(count,3,2,5)

            gate_kl=behavior_kl=0.
            kl_stops=0
            gate_loss=behavior_loss=0.
            # Schedule first, behavior second: independent parameter blocks.
            for _ in range(args.epochs):
                gl=replay(model.actor.scheduler,0).logsumexp(dim=-1)
                distribution=Categorical(logits=gl)
                gate_kl=float(categorical_kl(old_gall[idx],distribution.logits).mean().detach())
                if gate_kl>args.target_kl:
                    kl_stops+=1
                    break
                ratio=(distribution.log_prob(gates[idx])-old_glog[idx]).exp()
                loss=clipped_loss(ratio,advantage[idx],args.clip)-.002*distribution.entropy().mean()
                schedule_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.scheduler.parameters(),.5)
                schedule_optimizer.step()
                gate_loss=float(loss.detach())
            with torch.no_grad():
                new_gate=Categorical(logits=replay(model.actor.scheduler,0).logsumexp(dim=-1))
                gate_logratio=new_gate.log_prob(gates[idx])-old_glog[idx]
                gate_kl=float(categorical_kl(old_gall[idx],new_gate.logits).mean())
            for _ in range(args.epochs):
                conditional=replay(model.actor.behavior,1)
                distribution=select_movement(conditional,gates[idx])
                cond_kl=categorical_kl(old_ball[idx],torch.log_softmax(conditional,dim=-1))
                behavior_kl=float((old_gall[idx].exp()*cond_kl).sum(-1).mean().detach())
                if gate_kl+behavior_kl>args.target_kl:
                    kl_stops+=1
                    break
                # ACR-style product importance ratio, gate block held fixed.
                ratio=(gate_logratio.detach()+distribution.log_prob(moves[idx])-old_blog[idx]).exp()
                loss=clipped_loss(ratio,advantage[idx],args.clip)-.001*distribution.entropy().mean()
                behavior_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.behavior.parameters(),.5)
                behavior_optimizer.step()
                behavior_loss=float(loss.detach())
            for _ in range(args.epochs):
                pred=model.critic(states).squeeze(-1)
                pvalue,svalue=model.costs(states)
                value_loss=(clipped_value_loss(pred,old_value,reward_target)
                            +clipped_value_loss(pvalue,old_p,ptarget)+clipped_value_loss(svalue,old_s,starget))
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_parameters,.5)
                value_optimizer.step()
            if finished_packets:
                packet_multiplier=float(np.clip(packet_multiplier+10*(np.mean(finished_packets)-args.packet_budget),0,1000))
                risk_multiplier=float(np.clip(risk_multiplier+20*(np.mean(finished_risks)-args.risk_budget),0,1000))
            record={'steps':total,'update':update,'seconds':time.perf_counter()-started,
                    'mean_step_reward':float(a['reward'].mean()),'gate_loss':gate_loss,'behavior_loss':behavior_loss,
                    'critic_loss':float(value_loss.detach()),'gate_kl':gate_kl,'behavior_kl_before_last_step':behavior_kl,
                    'kl_early_stops':kl_stops,'packet_multiplier':packet_multiplier,'risk_multiplier':risk_multiplier,
                    'completed_episode_packet_ratio':float(np.mean(finished_packets)),
                    'completed_episode_collision_fraction':float(np.mean(finished_risks)),
                    'sampled_active_step_packet_ratio':float(a['packet'].sum()*600/(6*count))}
            if update%args.eval_every==0 or update==1 or total>=args.steps:
                validation=validate(model,**validation_args)
                record['validation']=validation
                score=selection_score(validation)
                if score>best:
                    best=score
                    save('best.pt',validation)
                save('last.pt',validation)
                status('running',validation)
                print(json.dumps(record),flush=True)
            log.write(json.dumps(record)+'\n')
            if (args.output/'STOP').exists():
                save('last.pt',validation)
                status('user_stop',validation)
                return
    status('budget_exhausted',validation)


if __name__=='__main__':
    main()
