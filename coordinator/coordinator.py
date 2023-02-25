import random
from typing import List, Set
import time
import math
from collections import deque
from flask import Flask, request
import redis
import os

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_DB = int(os.environ["REDIS_DB"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
PORT = os.environ.get("PORT", 5005)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)

class Bucket:
    def __init__(self, time: int):
        self.time = time
        self.count = 0
    def incr(self):
        self.count += 1

class RollingCounter: # this should probably be something we serialize in Redis but fine for now
    def __init__(self, granularity, window):
        self.window = window
        self.granularity = granularity
        self.buckets = deque()
        self.total = 0
    def observe(self):
        curr_time = math.floor(time.time_ns()/1e6) // self.granularity
        self.total += 1
        if len(self.buckets) > 0 and self.buckets[-1].time == curr_time:
            self.buckets[-1].incr()
        else:
            while len(self.buckets) > 0 and curr_time - self.buckets[0].time > self.window:
                b = self.buckets.popleft()
                self.total -= b.count
            self.buckets.append(Bucket(curr_time))
            self.buckets[-1].incr()
    def window_total(self):
        curr_time = math.floor(time.time_ns()/1e6) // self.granularity
        while len(self.buckets) > 0 and curr_time - self.buckets[0].time > self.window:
            b = self.buckets.popleft()
            self.total -= b.count
        return self.total
    
counters = {}

REQUESTS_PER_HOST = 25

def handle_model_call(model: str):
    if not model in counters:
        counters[model] = RollingCounter(1000, 10) # 1000ms buckets, keep 10
    counters[model].observe()
    target_hosts = math.ceil(counters[model].window_total()/10/REQUESTS_PER_HOST)
    model_hosts = r.scard(f'model:{model}:shard')
    total_hosts = r.scard('hosts')
    if target_hosts == model_hosts:
        return
    elif target_hosts < model_hosts:
        for _ in range(model_hosts - target_hosts):
            r.srem(f'model:{model}:shard', r.srandmember(f'model:{model}:shard'))
    else:
        available = set(r.sdiff('hosts', f'model:{model}:shard')) # not the most efficient, but functional for now
        for _ in range(min(len(available), target_hosts-model_hosts)):
            elem = random.sample(available, 1)[0]
            available.remove(elem)
            r.sadd(f'model:{model}:shard', elem)

if __name__ == "__main__":
    sub = r.pubsub()
    sub.subscribe('inference')
    for message in sub.listen():
        model = message.get('data')
        handle_model_call(model)