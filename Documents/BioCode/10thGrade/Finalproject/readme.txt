Hello, 
if you dont like english then I probably wrote an hebrew manual in the final docs.

Before you run the program if you wish there are 
some paramaters that you are able to control.
check ./game/config.py and edit the params in there.

in order to run the program the terminal is used.
there is a main program wrapper of sorts main.py 
that is used to run all options of the program.

to begin type in the terminal: python main.py

there will be 6 running options:

- 1 simulation demo:
    a basic run with the current params you set and the base delievered models.

-2 train prey:
    not relevant unless you wish to individually train a prey that will not be used with the predator ai.

-3 train predator:
    not relevant unless you wish to individually train a predator that will not be used with the prey ai.

-4 joint-train both:
    the generally used training option that will train both simultaneously against one another

-5 two-phase:
    another used training option for more optimal training where you train the predator and prey individually one after the other.

-6 showcase:
    this is the main showcase for the program, it will check if there is already collected data, otherwise it will run 1000 runs by default and collect
    data afterwhich it will present it via graphs as a form of presentation.

****NOTICE****
for most options there will be other sub-options or sub-paramaters to select, generally it is very straightforward and self explainable,
mostly just pressing enter will make it go with the default settings which is recommended.