import problem_1
import problem_2

problem_1.verbose = False
problem_1.Problem1().solve()
input('End of solution 1. Press Enter to continue...')

problem_2.verbose = True
problem_2.Problem2().solve()
input('End of solution 2. Press Enter to continue...')

with open('problem_3.py') as f:
    code = compile(f.read(), 'problem_3.py', 'exec')
    exec(code)
input('End of solution 3. Press Enter to continue...')