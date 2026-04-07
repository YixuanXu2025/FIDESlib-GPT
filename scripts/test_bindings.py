from hegpt import FidesContext

x = [1.0, 2.0, 3.0, 4.0, 0.5, -1.25, 8.0, 9.5]
y = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

with FidesContext(devices=[0]) as ctx:
    print("info:", ctx.info())
    print("roundtrip:", ctx.roundtrip(x))
    print("add:", ctx.add(x, y))
    print("mult_scalar:", ctx.mult_scalar(x, 2.5))
