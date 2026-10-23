"""Bounded, measured work-rate estimates shared by training and retrieval.

No work count is guessed. ETA stays unknown until enough distinct samples exist.
"""
from collections import deque
import statistics
import time

class WorkProgress:
    def __init__(self,clock=time.monotonic,window=20):
        self.clock=clock;self.samples=deque(maxlen=window);self.started=clock();self.done=0
        self.state='queued'
    def update(self,done,total=None,state='running'):
        now=self.clock()
        if done<self.done:raise ValueError('work cannot move backwards within a task')
        self.done=done;self.state=state
        if not self.samples or done>self.samples[-1][1]:self.samples.append((now,done))
        rates=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(self.samples,list(self.samples)[1:]) if b[0]>a[0]]
        rate=statistics.median(rates) if len(rates)>=3 else None
        left=max(0,total-done) if total is not None else None
        eta=left/rate if rate and left is not None and state=='running' else None
        return {'state':state,'completed':done,'total':total,'elapsedSeconds':now-self.started,
            'unitsPerSecond':rate,'etaSeconds':eta,'etaLowerSeconds':eta*.7 if eta else None,
            'etaUpperSeconds':eta*1.4 if eta else None}
