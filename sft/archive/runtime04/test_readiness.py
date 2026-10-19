import copy, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))

class ReadinessTests(unittest.TestCase):
    def fixture(self):
        import controller as c
        root=Path(self.tmp.name);out=root/'out';out.mkdir();data=root/'data';data.mkdir()
        files={name:{'bytes':12,'sha256':'a'*64} for name in sorted(c.FILES)}
        manifest=data/'manifest.json';c.write(manifest,{'files':files,'train_dialogues':4369280,'heldout_dialogues':45090,'sequence_length':16384})
        producer=root/'producer-freeze.json';c.write(producer,{})
        benchmark=root/'results/midtraining/offline2b-ablation-gpu-01/parent-final-acceptance.json';c.write(benchmark,{'fixture':True})
        freeze=out/'source-freeze.json';c.write(freeze,{})
        ready=dict(schema='nemochat-sealed-full03-v1',status='accepted_sealed_snapshot_full03',snapshot_id='fixture',sft_manifest_sha256=c.sha(manifest),c_manifest_sha256='b'*64,holder_identity={'pid':os.getpid(),'starttime':123,'uid':os.getuid()},holder_unit='nemochat-sft-sealed03-holder.service',socket_path=str(root/'private/socket'),files={},audits={'fixture':True},provenance={'fixture':True},producer_source_freeze_sha256=c.sha(producer),cpu_consumer_receipt={'fixture':True})
        for name,v in files.items():ready['files']['sft:'+name]=dict(v,file_id='sft:'+name,seals=15,access_mode='O_RDONLY',shape=None,dtype=None)
        for split,n in [('train',4369280),('heldout',45090)]:
            for name,shape,dtype in [('tokens',[n,16385],'uint32'),('masks',[n,16384],'uint8')]:ready['files'][f'sft:{split}-{name}.npy'].update(shape=shape,dtype=dtype)
            fid='rowhash:'+split;ready['files'][fid]=dict(file_id=fid,bytes=n*32,sha256='c'*64,seals=15,access_mode='O_RDONLY',shape=[n,32],dtype='uint8')
        cm={'sources':{source:{split:{'bin_sha256':'b'*64,'idx_sha256':'b'*64,'bin_bytes':4,'idx_bytes':4} for split in ('train','validation')} for source in ('code_concepts','algorithmic','formal_logic','climbmix_replay')}}
        cmp=root/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json';c.write(cmp,cm);ready['c_manifest_sha256']=c.sha(cmp)
        for source,splits in cm['sources'].items():
            for split in splits:
                for suffix in ('bin','idx'):
                    fid=f'c:{source}/{split}/corpus.{suffix}';ready['files'][fid]=dict(file_id=fid,bytes=4,sha256='b'*64,seals=15,access_mode='O_RDONLY',shape=None,dtype=None)
        p=root/'READY.sealed.json';c.write(p,ready)
        release=dict(schema='nemochat-sealed-runtime04-parent-release-v1',status='parent_accepted_runtime04',input_mode='sealed_scm_rights',source_hf=c.SOURCE,source_freeze_sha256=c.sha(freeze),ready_path=str(p),ready_sha256=c.sha(p),snapshot_id='fixture',holder_identity=ready['holder_identity'],producer_source_freeze_path=str(producer),producer_source_freeze_sha256=c.sha(producer),benchmark_acceptance_sha256=c.sha(benchmark),checkpoint_integrity='not_checked_explicit_user_waiver',automatic_main_after_calibration=False,normalization={'ready_sha256':c.sha(p),**{key:True for key in ['all7_original_sft_exact_sha','all16_c_exact_sha','fresh_c_reserved_id_absence','sft_masks_bounds_target_counts','sft_alias18_29_audited','tokenizer_provenance','pinned_cpu_same_fd_all7_hash','readonly_sealed_descriptors']}})
        proof=dict(status='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS',ready_sha256=c.sha(p),snapshot_id='fixture',holder_identity=ready['holder_identity'],runtime_source_freeze_sha256=c.sha(freeze),container_id='fixture',ranks=[dict(rank=rank,producer_client='REAL_PUBLIC_CLIENT',received_fds=7,ready_sha256=c.sha(p),status='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS') for rank in range(8)])
        c.write(out/'actual-sealed-cpu-proof.json',proof);release['actual_cpu_adapter_receipt_sha256']=c.sha(out/'actual-sealed-cpu-proof.json')
        guard=out/'config-sdk03/rank-0-source-guard.json';c.write(guard,dict(unchanged_sdk_guard_invoked=True,local_wrapper_identity_verified=True,runtime_source_freeze_sha256=c.sha(freeze)))
        regression=out/'config-sdk03/source-guard-regression.json';c.write(regression,dict(status='PINNED_ACTUAL_SOURCE_GUARD_REGRESSIONS_PASS',negative_controls={str(i):'DENIED' for i in range(5)}))
        release.update(actual_source_guard_receipt_sha256=c.sha(guard),source_guard_regression_sha256=c.sha(regression))
        rp=root/'release.json';c.write(rp,release)
        return c,root,out,manifest,p,ready,rp,release
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(dir='/home/yoon/.hermes/cache/scratch')
    def tearDown(self):self.tmp.cleanup()
    def test_sealed_release_and_fail_closed_mutations(self):
        import readiness as g
        c,root,out,m,p,ready,rp,r=self.fixture()
        with patch.multiple(c,ROOT=root,OUT=out,MANIFEST=m,MANIFEST_SHA=c.sha(m)),patch.object(c,'source_paths',return_value=[]),patch.object(g,'live_holder',return_value=True),patch.object(g,'C_MANIFEST_SHA',ready['c_manifest_sha256']),patch.object(c,'engine') as engine:
            engine.return_value.request.return_value={'Id':'fixture','Name':'/nemochat-sft-sealed03-r04-actual-cpu','Config':{'Image':c.IMAGE,'Labels':{c.OWNER:'actual-sealed-cpu','nemochat.sealed03':'v1'}},'HostConfig':{'DeviceRequests':[]},'State':{'Running':False,'ExitCode':0,'OOMKilled':False}}
            self.assertEqual(c.validate_release(rp)['input_mode'],'sealed_scm_rights')
            for key,value in [('input_mode','buffered_mmap'),('snapshot_id','wrong'),('holder_identity',{}),('source_hf','BASE'),('status','parent_accepted_runtime02'),('producer_source_freeze_sha256','d'*64),('ready_sha256','e'*64)]:
                with self.subTest(key=key):
                    bad=dict(r,**{key:value});c.write(rp,bad)
                    with self.assertRaises(PermissionError):c.validate_release(rp)
            c.write(rp,r)
            for key in r['normalization']:
                if key=='ready_sha256':continue
                bad=copy.deepcopy(r);bad['normalization'][key]=False;c.write(rp,bad)
                with self.subTest(key=key),self.assertRaises(PermissionError):c.validate_release(rp)
            c.write(rp,r)
            with patch.object(g,'live_holder',side_effect=PermissionError('dead holder')):
                with self.assertRaises(PermissionError):c.validate_release(rp)
    def test_real_live_identity_stale_pid_and_socket_type(self):
        import readiness as g
        import controller as c
        import socket,types
        root=Path(self.tmp.name);directory=root/'private';directory.mkdir();sock=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);sock.bind(str(directory/'socket'))
        identity=c.load_module('sealed03_real_identity',c.ROOT/'sft/training/controls.py').process_identity(os.getpid())
        ready={'holder_identity':identity,'holder_unit':'nemochat-sft-sealed03-holder.service','socket_path':str(directory/'socket')}
        try:
            with patch.object(g.subprocess,'run',return_value=types.SimpleNamespace(stdout=f'MainPID={os.getpid()}\nActiveState=active\n')):
                self.assertTrue(g.live_holder(ready,c))
                ready['holder_identity']=dict(identity,starttime=identity['starttime']+1)
                with self.assertRaises(PermissionError):g.live_holder(ready,c)
                ready['holder_identity']=identity;regular=root/'not-socket';regular.write_text('fixture');ready['socket_path']=str(regular)
                with self.assertRaises(PermissionError):g.live_holder(ready,c)
        finally:sock.close()

    def test_gpu_payload_owned_label_and_exact_socket_mount(self):
        import controller as c
        with patch.object(c,'socket_bind',return_value=['/fixture/private:/fixture/private:ro']):
            payload=c.container_payload('calibration',Path('/out/config'),1000)
        self.assertEqual(payload['Labels']['nemochat.sealed03'],'v1')
        self.assertEqual(payload['HostConfig']['Binds'][-1],'/fixture/private:/fixture/private:ro')
        self.assertEqual(payload['HostConfig']['DeviceRequests'][0]['DeviceIDs'],list(map(str,range(8))))
        c.check_owned({'Id':'id','Name':'/nemochat-sft-sealed03-r04-calibration','Config':{'Labels':payload['Labels']}},'id','calibration')
        payload['Labels'].pop('nemochat.sealed03')
        with self.assertRaises(PermissionError):c.check_owned({'Id':'id','Name':'/nemochat-sft-sealed03-r04-calibration','Config':{'Labels':payload['Labels']}},'id','calibration')

if __name__=='__main__':unittest.main()
