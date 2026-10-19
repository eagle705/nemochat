#!/usr/bin/env python3
"""Parent binds fresh READY after review; no wait loop and no GPU launch."""
import argparse,json
from pathlib import Path
import controller as c
import contract
from readiness import validate_ready,validate_parent_release

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--ready',type=Path,required=True);p.add_argument('--producer-freeze',type=Path,required=True)
 p.add_argument('--parent-accept-fresh-transport-and-identical-data-audits',action='store_true',required=True)
 a=p.parse_args();ready=validate_ready(c.read(a.ready),c,True)
 r=dict(schema='nemochat-native905-continue1500-parent-release-v1',status='parent_accepted_continue1500',input_mode='sealed_scm_rights',resume_checkpoint=contract.CHECKPOINT,full_state_resume=True,manifest_sha256=c.MANIFEST_SHA,checkpoint_integrity='not_checked_explicit_user_waiver',source_freeze_sha256=c.sha(c.OUT/'source-freeze.json'),ready_path=str(a.ready.resolve()),ready_sha256=c.sha(a.ready),snapshot_id=ready['snapshot_id'],holder_identity=ready['holder_identity'],producer_source_freeze_path=str(a.producer_freeze.resolve()),producer_source_freeze_sha256=ready['producer_source_freeze_sha256'],parent_data_transport_received=True,parent_identical_authenticated_data_audits_accepted=True,source_guard_receipt_sha256=c.sha(c.OUT/'config-sdk03/actual-post-wrapper-guard.json'),source_guard_regression_sha256=c.sha(c.OUT/'config-sdk03/source-guard-regression.json'))
 path=c.OUT/'parent-release.json';c.require(not path.exists(),'do not overwrite parent release');c.write(path,r);validate_parent_release(path,c);print(json.dumps({'status':'PARENT_RELEASE_BOUND_NOT_STARTED','path':str(path),'sha256':c.sha(path)},indent=2))
if __name__=='__main__':main()
