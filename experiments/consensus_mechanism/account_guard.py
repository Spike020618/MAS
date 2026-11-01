"""Stop the whole batch after a persisted account refusal, rather than score fallback."""
import json
class AccountUnavailable(Exception):pass

def check(raw):
 status=raw.get('status_code')
 body=str(raw.get('body','')).lower()
 if status in (401,402,403)or(status==400 and any(s in body for s in ('overdue','good standing','insufficient','arrear','access denied'))):
  raise AccountUnavailable('Provider account unavailable; raw receipt retained; batch stopped. Resolve account before using a NEW run directory.')

def install(provider):
 parser=provider.parser
 def guarded_parser(raw,request):
  check(raw)
  return parser(raw,request)
 provider.parser=guarded_parser
 transport=provider.transport;preflight=transport.preflight
 def guarded_preflight(request):
  for f in transport.slots.glob('*.raw.json'):
   row=json.loads(f.read_text());check(row.get('raw',{}))
  return preflight(request)
 transport.preflight=guarded_preflight
