from .schema import digest

def merge(reviews):
    # Preserve conflicting observations; no sequential answer overwrites.
    evidence=[];seen=set()
    for review in reviews:
        if not review.valid:continue
        for observation in review.observations:
            key=digest([observation,review.citation_ids])
            if key not in seen:evidence.append({'text':observation,'source_ids':list(review.citation_ids),'action_id':review.action_id});seen.add(key)
    return evidence
