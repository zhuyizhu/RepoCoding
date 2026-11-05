"""Read-only local readiness audit. Does not contact any model or print secrets."""
from pathlib import Path
import json,shutil,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
def main():
    checks={}
    for name in ('git','docker','node','npm'):
        checks[name]=bool(shutil.which(name))
    inputs={'codelens':ROOT/'runtime/codelens/agent/agents/pipeline.py','evistate':ROOT/'runtime/evistate/mechanism2/core/tracker.py',
        'repograph':ROOT.parent/'dubbo/3.1-d0a1bd0/graph.pkl','tags':ROOT.parent/'dubbo/3.1-d0a1bd0/tags.json','java_pom':ROOT.parent/'repositories/apache__dubbo/pom.xml'}
    checks.update({name:path.is_file() for name,path in inputs.items()})
    repo=ROOT.parent/'repositories/apache__dubbo'
    if checks['git'] and checks['java_pom']:
        checks['revision']=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        checks['clean_checkout']=not subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
    if checks['docker']:
        checks['docker_engine']=subprocess.run(['docker','info','--format','{{.ServerVersion}}'],capture_output=True,text=True,timeout=15).returncode==0
    checks['env_file_present']=(ROOT/'.env').is_file()
    checks['model_probe']='not attempted; user will start Qwen later'
    print(json.dumps(checks,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
