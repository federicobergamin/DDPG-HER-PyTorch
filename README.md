## Experiments with RL

Code I wrote in 2018 where I implemented the following papers

1. *"Continuous control with deep reinforcement learning"* by Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. 

2. *"Hindsight Experience Replay"* by Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, O.P. and Zaremba, W.

UPDATE (2020): TODO: I noticed that there is a small error in my HER implementation. In the original paper the input they used is the goal postion - current position, while in my case I used only the current position. Since my MuJoCo licence expired, I leave it as a TODO.