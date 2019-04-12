import re as regex
import difflib
from sympy import *
from sympy.matrices import Matrix
from sympy.matrices.expressions.matexpr import MatrixElement
from itertools import combinations
import sys
from copy import copy, deepcopy
import csv


At2 = ''
At = ''
Bt2 = ''
Bt = ''
Gg = ''
Gg2 = ''

winograd_alpha = None
winograd_m = None
winograd_r = None


def multiply(matrix_a, matrix_b):
    res = matrix_a*matrix_b
    return res

def __find_diffs(str1, str2):
    delta = list()
    if len(str1) == len(str2):
        for idx, c in enumerate(zip(str1, str2)):
            if c[0] == c[1]:
                continue
            elif c[0].isdigit() and c[1].isdigit():
                if int(c[0]) < int(c[1]):
                    delta.append(idx)
            else:
                exit("Error: Not reducible matrices. No integer.")

    return delta

# def __find_diffs(str1, str2):
#     delta = list()
#     for diff in difflib.SequenceMatcher(None, str1, str2).get_opcodes():
#         # diff = ('op', str1.diff.begin, str2.diff.end, str2.diff.begin, str2.diff.end)
#         if diff[0] == 'equal': continue
#         elif diff[0] == 'replace':
#             if diff[2]-diff[1] == 1 and diff[4]-diff[3] == 1:
#                 try:
#                     if int(str1[diff[1]]) < int(str2[diff[3]]):
#                         delta.append(diff[1])
#                 except:
#                     exit('Error: Not reducible matrices. No integer.')
#             else:
#                 print diff
#                 print str1
#                 print str2
#                 print (str1[diff[1]:diff[2]], str2[diff[1]:diff[2]]) 
#                 exit('Error: Not reducible matrices. Not a single character diff.')
#         else:
#             exit('Error: Not reducible matrices. Terms do not match. (insert difflib opcode)')
                
#     return delta

def __replace_term(sample_1, sample_2):
    indices = __find_diffs(sample_1, sample_2)
    if indices: # check not empty
        replica = list(sample_1)
        for index in indices:
            replica[index] = '~'
        replica = ''.join(replica)
        return replica.replace('Integer(~)', "Symbol('i')")

def eliminate_ones(matrix):
    for idx, row in enumerate(matrix):
        poly = 0
        if row.func == Add:
            for arg in row.args:
                coeff, symbol = arg.args
                if abs(coeff) == 1.0:
                    poly = poly + int(coeff)*symbol
                else:
                    poly = poly + coeff*symbol
        elif row.func == Mul:
            coeff, symbol = row.args
            if abs(coeff) == 1.0:
                poly = poly + int(coeff)*symbol
            else:
                poly = poly + coeff*symbol
        
        matrix[idx] = poly

    return matrix


def partial_factor(row):
    out = 0
    groups = {}
    terms_coeffs = row.as_coefficients_dict()
    for symbol, coeff in terms_coeffs.iteritems():
        if abs(coeff) in groups.keys():
            groups[abs(coeff)].append(Mul(symbol, coeff))
        else:
            groups[abs(coeff)] = [Mul(symbol, coeff)]

    for group in groups.itervalues():
        out += factor(Add(*group))

    return out


def __find_cse(matrix, symbol_start_count=0):
    small_vars_map = {}
    resulting_vars = []
    variables, refactored_rows = cse(matrix, symbols=numbered_symbols('x', start=symbol_start_count))
    refactored_rows = refactored_rows[0]
    cleared_vars =[]
    for var in variables:
        if len(var[1].free_symbols) < 2:
            small_vars_map[var[0]] = var[1]
        else:
            cleared_vars.append(var)

    for var in cleared_vars:
        var = [var[0], var[1].subs(small_vars_map)]
        resulting_vars.append(var)

    refactored_rows = refactored_rows.subs(small_vars_map)

    return {
        'vars':resulting_vars,
        'exprs': refactored_rows
    }

def extract_common_terms(matrix):
    mat, symbol_map_1 = __matrixSymbols_to_normalSymbols(matrix)
    mat_1 = __find_common_terms(mat) #contains exprs and vars
    mat_cse_res = __normalSymbols_to_matrixSymbol(mat_1, symbol_map_1)

    symbols_mat = Matrix([term[1] for term in mat_cse_res['vars']])
    smat, symbol_map_2 = __matrixSymbols_to_normalSymbols(symbols_mat, 'y')
    mat_2 = __find_common_terms(smat, len(mat_1['vars']))
    symbols_cse_res = __normalSymbols_to_matrixSymbol(mat_2, symbol_map_2)

    for vidx, var in enumerate(mat_cse_res['vars']):
        mat_cse_res['vars'][vidx][1] = symbols_cse_res['matrix'][vidx]
    mat_cse_res['vars'].extend(symbols_cse_res['vars'])

    return mat_cse_res

def __find_common_terms(reduced_matrix, symbol_start_count=0):
    mat = reduced_matrix
    multiply_by_negative_idx_list = []
    for idx, row in enumerate(mat):
        first_try = __find_cse(mat)
        mat[idx] = -mat[idx]
        second_try = __find_cse(mat)
        if len(second_try['vars']) <= len(first_try['vars']):
             mat[idx] = -mat[idx]
        else:
            multiply_by_negative_idx_list.append(idx)

    result = __find_cse(mat, symbol_start_count)
    for i in multiply_by_negative_idx_list:
        result['exprs'][i] = -result['exprs'][i]

    return result

def __matrixSymbols_to_normalSymbols(matrix, symbol_letter='z'):
    cnt = 0
    reduced_symbol_map = {}
    for idx, row in enumerate(matrix):
        terms = row.as_coefficients_dict()
        for term_symbol in terms.iterkeys():
            if term_symbol not in reduced_symbol_map:
                reduced_symbol_map[term_symbol] = sympify('{0}{1}'.format(symbol_letter, cnt))
                cnt = cnt+1
 
    matrix = matrix.subs(reduced_symbol_map)
    return matrix, reduced_symbol_map

def __normalSymbols_to_matrixSymbol(reduced_matrix, symbol_map):
    inv_map = {v: k for k, v in symbol_map.iteritems()}
    mat_vars = []
    mat = reduced_matrix['exprs']
    for key, value in symbol_map.iteritems():
        if key.count_ops() == 0:
            mat = mat.replace(value, key)
        else:
            mat_vars.append([value, key])

    for var in reduced_matrix['vars']:
        mat_vars.append([var[0], var[1].subs(inv_map)])
    
    return {'matrix': mat,
            'vars': mat_vars}


def apply_partial_factor(matrix):
    for idx, row in enumerate(matrix):
        matrix[idx] = partial_factor(matrix[idx])

    return matrix


def generate_fma(reduced_matrix):
    def __get_mults_symbols(row):
        mults = list()
        symbols = list()
        for term in row.args:
            if type(term) == Symbol or type(term) == MatrixElement:
                symbols.append(term)
            elif type(term) == Mul and -1 in term.args:
                symbols.append(term)
            elif type(term) == Mul:
                mults.append(term)

        return mults, symbols

    def __replace_with_fma(row):
        if row.func == Add:
            mults, symbols = __get_mults_symbols(row)
            for symbol, mul in zip(symbols, mults):
                a,b = mul.args
                c = symbol
                if -1 not in (a,b):
                    fma_term = sympify('fma({0},{1},{2})'.format(a,'tmpb', 'tmpc'))
                    fma_term = fma_term.subs({sympify('tmpb'): b, sympify('tmpc'): c})
                    row = row.subs(mul+symbol, fma_term)
        return row

    for idx, row in enumerate(reduced_matrix['matrix']):
        reduced_matrix['matrix'][idx] = __replace_with_fma(row)

    for idx, row in enumerate(reduced_matrix['vars']):
        reduced_matrix['vars'][idx][1] = __replace_with_fma(row[1])

    return reduced_matrix


def reduce_terms(input, direction):
    unroll_max = 0
    if direction == 'row':
        unroll_max = input.shape[1]
        reduced_matrix = zeros(input.shape[0], 1)
        for i in range(input.shape[0]):
            sample_1, sample_2 = srepr(input[i,0]), srepr(input[i,1])
            reduced_matrix[i] = eval(__replace_term(sample_1, sample_2))

    elif direction == 'col':
        unroll_max = input.shape[0]
        reduced_matrix = zeros(1, input.shape[1])
        for i in range(input.shape[1]):
            sample_1, sample_2 = srepr(input[0,i]), srepr(input[1,i])
            reduced_matrix[i] = eval(__replace_term(sample_1, sample_2))
    else:
        exit('Error: Unspecified major order.')

    reduced_matrix = eliminate_ones(reduced_matrix)
    mat = deepcopy(reduced_matrix)
    advanced_reduced_matrix = apply_partial_factor(mat)
    advanced_reduced_matrix = extract_common_terms(advanced_reduced_matrix)
    advanced_reduced_matrix = generate_fma(advanced_reduced_matrix)


    return {'fully_reduced': (advanced_reduced_matrix['matrix'], advanced_reduced_matrix['vars']),
            'partial_reduced': reduced_matrix,
            'unroll_max_count': unroll_max
            }



def arithmetic_calculator(profile, name):
    def __count_adds(row):
        return str(row).count(' - ') + str(row).count(' + ')# + str(row).count('fma')

    def __count_muls(row):
        return str(row).count('*')# + str(row).count('fma')

    def __count_fmas(row):
        return str(row).count('fma')
        
    before_mults = 0
    before_adds = 0
    after_mults = 0
    after_adds = 0
    fma_count = 0
    for row in profile['partial_reduced']:
        before_mults += __count_muls(row)
        before_adds += __count_adds(row)

    for row in profile['fully_reduced'][0]:
        after_mults += __count_muls(row)
        after_adds += __count_adds(row)
        fma_count += __count_fmas(row)
    for var in profile['fully_reduced'][1]:
        after_mults += __count_muls(var[1])
        after_adds += __count_adds(var[1])
        fma_count += __count_fmas(var[1])

    # print "Before:"
    # for row in matrix_before:
    #     print row
    # print "After:"
    # for row in matrix_after[0]:
    #     print row
    # for row in matrix_after[1]:
    #     print row
    print ">> {name}: Multiplies = {after_mults}/{before_mults}, Additions = {after_adds}/{before_adds}, FMAs = {fma_count}, unroll_count = {unroll_count}\n".format(**{
        'name': name,
        'after_mults': after_mults*profile['unroll_max_count'],
        'before_mults': before_mults*profile['unroll_max_count'],
        'after_adds': after_adds*profile['unroll_max_count'],
        'before_adds': before_adds*profile['unroll_max_count'],
        'fma_count': fma_count*profile['unroll_max_count'],
        'unroll_count': profile['unroll_max_count']
        })

    with open('results.csv', mode='a') as results_fd:
        results_fd = csv.writer(results_fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_fd.writerow(['F({m},{r})'.format(m=winograd_m, r=winograd_r),
                             winograd_m,
                             winograd_r,
                             winograd_alpha,
                             name,
                             after_mults*profile['unroll_max_count'],
                             before_mults*profile['unroll_max_count'],
                             after_adds*profile['unroll_max_count'],
                             before_adds*profile['unroll_max_count'],
                             fma_count])


def generate_code(matrix, var_name, order, loop_iteration, unroll_factor=1):
    def __find_best_unroll_factor(loop_iteration, unroll_factor):
        if unroll_factor >= loop_iteration:
            return loop_iteration
        elif not loop_iteration%unroll_factor == 0:
            for i in range(unroll_factor, loop_iteration+1):
                if loop_iteration%i == 0:
                    return i
        else:
            return unroll_factor

    unroll_factor = __find_best_unroll_factor(loop_iteration, unroll_factor)
    full_unroll = True if unroll_factor == loop_iteration else False

    output_str = []

    output_str.append("for (int i = 0; i < {0}; i+={1}) {{".format(loop_iteration, unroll_factor))
    for unroll_count in range(unroll_factor):
        for row in reversed(matrix[1]):
            value = regex.sub(r'\[(.+?), (.+?)\]', r'[\1][\2]', str(row[1]))
            value = regex.sub(r'\[(i)\]', r'[\1+{0}]'.format(unroll_count), value)
            value = regex.sub(r'([xz]\d+)', r'\1_{0}'.format(unroll_count), value)
            value = regex.sub(r'fma\((.+?),', r'fma((float)\1,', value)
            output_str.append("  float {var}_{unrollc} = {value};".format(var=row[0], unrollc=unroll_count, value=value))

        for idx, row in enumerate(matrix[0]):
            if order == 'row':
                template = "  {var}[{index}][i+{unrollc}] = {value};"
            else:
                template = "  {var}[i+{unrollc}][{index}] = {value};"

            value = regex.sub(r'\[(.+?), (.+?)\]', r'[\1][\2]', str(row))
            value = regex.sub(r'\[(i)\]', r'[\1+{0}]'.format(unroll_count), value)
            value = regex.sub(r'([xz]\d+)', r'\1_{0}'.format(unroll_count), value)
            value = regex.sub(r'fma\((.+?),', r'fma((float)\1,', value)
            output_str.append(template.format(var=var_name, index=idx, value=value, unrollc=unroll_count))

    output_str.append("}\n")

    # remove loop related syntaxes if full unrolling is selected
    if full_unroll:
        output_str = output_str[1:-1] # remove for loop block
        for idx, i in enumerate(output_str):
            output_str[idx] = i.replace('i+','')

    output_res = '\n'.join(output_str)
    output_res = output_res.replace('i+0', 'i') # pruning i+0
    return output_res


def main(pat, pg, pbt, unroll_factor=1):
    global At2
    global At
    global Bt2
    global Bt
    global Gg
    global Gg2

    global winograd_alpha
    global winograd_r
    global winograd_m

    winograd_alpha = pbt.shape[0]
    winograd_r = pg.shape[1]
    winograd_m = pat.shape[0]

    print "F(m:{m}, r:{r}), alpha = {alpha}\n".format(m=winograd_m, r=winograd_r, alpha=winograd_alpha)

    G_matrix = Matrix(pg)
    # print("anchor")
    r = G_matrix.shape[1]
    g_matrix = Matrix(MatrixSymbol('g', r, r))

    At_matrix = Matrix(pat);
    # print(At_matrix.shape)
    a = At_matrix.shape[1]
    m_matrix = Matrix(MatrixSymbol('m', a, a))

    Bt_matrix = Matrix(pbt);
    d_matrix = Matrix(MatrixSymbol('d', a, a))


    Gg_res = multiply(G_matrix, g_matrix)

    GgGt_res = multiply(Matrix(MatrixSymbol('Gg', Gg_res.shape[0], Gg_res.shape[1])), G_matrix.T)

    Btd_res = multiply(Bt_matrix, d_matrix)

    BtdB_res = multiply(Matrix(MatrixSymbol('Btd', Btd_res.shape[0], Btd_res.shape[1])), Bt_matrix.T)

    Atm_res = multiply(At_matrix, m_matrix)

    AtmA_res = multiply(Matrix(MatrixSymbol('Atm', Atm_res.shape[0], Atm_res.shape[1])), At_matrix.T)

    

    Gg = ''
    Gg_reduced = reduce_terms(Gg_res, 'row')
    test = Gg_reduced['fully_reduced']
    Gg = generate_code(test, 'Gg', 'row', Gg_reduced['unroll_max_count'], unroll_factor)
    
    print(Gg)
    arithmetic_calculator(Gg_reduced, 'Gg')


    Gg2 = ''
    GgGt_reduced = reduce_terms(GgGt_res, 'col')
    test = GgGt_reduced['fully_reduced']
    Gg2 = generate_code(test, 'tmp', 'col', GgGt_reduced['unroll_max_count'], unroll_factor)
    
    print(Gg2)
    arithmetic_calculator(GgGt_reduced, 'GgGt')



    Bt = ''
    Bt_reduced = reduce_terms(Btd_res, 'row')
    test = Bt_reduced['fully_reduced']
    Bt = generate_code(test, 'Btd', 'row', Bt_reduced['unroll_max_count'], unroll_factor)
    
    print(Bt)
    arithmetic_calculator(Bt_reduced, 'Btd')


    Bt2 = ''
    Bt2_reduced = reduce_terms(BtdB_res, 'col')
    test = Bt2_reduced['fully_reduced']
    Bt2 = generate_code(test, 'd', 'col', Bt2_reduced['unroll_max_count'], unroll_factor)
    
    print(Bt2)
    arithmetic_calculator(Bt2_reduced, 'BtdB')


    At = ''
    At_reduced = reduce_terms(Atm_res, 'row')
    test = At_reduced['fully_reduced']
    At = generate_code(test, 'Atm', 'row', At_reduced['unroll_max_count'], unroll_factor)
    
    print(At)
    arithmetic_calculator(At_reduced, 'Atm')


    At2 = ''
    At2_reduced = reduce_terms(AtmA_res, 'col')
    test = At2_reduced['fully_reduced']
    At2 = generate_code(test, 'tmp', 'col', At2_reduced['unroll_max_count'], unroll_factor)
    
    print(At2)
    arithmetic_calculator(At2_reduced, 'AtmA')

    return At2

def getAt():
    return At
def getAt2():
    return At2
def getBt():
    return Bt
def getBt2():
    return Bt2
def getGg():
    return Gg 
def getGg2():
    return Gg2

