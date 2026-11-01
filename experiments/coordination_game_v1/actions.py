from .schema import Action

def candidates(members,tasks,source_ids):
    # Shared rights. Local history is a feature, not a restriction on source access.
    return tuple(Action(f'{m.member_id}/{t.target_id}/{s}',m.member_id,t.target_id,(s,))for m in members for t in tasks for s in sorted(source_ids))
