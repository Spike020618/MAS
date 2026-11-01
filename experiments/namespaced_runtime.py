"""Separate logical method identity from shared policy/physical request identity."""
class NamespacedAPI:
    def __init__(self,api,namespace):
        if not namespace or '/'in namespace or '\\'in namespace:raise ValueError('simple nonempty namespace required')
        self.api=api;self.namespace=namespace
    def call(self,slot,request):return self.api.call(self.namespace+'__'+slot,request)
