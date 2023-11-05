from gym import spaces

space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8 #참이 아닐 경우 에러뜨는 코드임