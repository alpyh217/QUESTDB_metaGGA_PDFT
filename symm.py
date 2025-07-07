# This library is to convert the symmetries from literature to those supported by MCSCF
# MCSCF Supported Symmetries: Cs, C2v, C2h, D2h

SUPPORTED_SYMS = ["Cs","C2v","C2h","D2h"]

def reduce_point_group(mol): # NOT USED
    """
    These point groups are supported by PySCF but not by mcscf
    So we need to explicitly reduce these molecular point groups to one supported by the mcscf solver
    This will be done automatically by PySCF for the other symmetries
    """

    pointgroup_trans = {
        'C3v':'Cs',
        'C4v':'C2v',
        'D3h':'C2v',
        'D4h':'D2h',
        'D6h':'D2h',
        'Coov':'C2v',
        'Dooh':'D2h',
        'Oh':'D2h',
        'Td':'C2v',
        'SO3':'D2h',
    }
    if mol.topgroup in pointgroup_trans.keys():
        mol.symmetry = pointgroup_trans[mol.topgroup]
        mol.build()
        print(f"Point group {mol.topgroup} reduced to {mol.groupname}")
    elif mol.topgroup in SUPPORTED_SYMS:
        pass
    else:
        raise ValueError(f"Point group {mol.topgroup} not supported in this function")
    return mol

def translate_symmetries(mol,syms): # NOT USED
    """
    Translates diatomic symmetries to the alternative notation readable by pyscf
    Note: Weird things can happen with the "-" symbol, etc. when string matching
    So just make sure things are consistent via copy/paste if you get an error here
    """
    
    sym_trans = {
    'Dooh':{"Sg+":"A1g",
            "Sg-":"A2g",
            "Pg":"E1g",
            "Dg":"E2g",
            "Fg":"E3g",
            "Su+":"A1u",
            "Su-":"A2u",
            "Pu":"E1u",
            "Du":"E2u",
            "Fu":"E3u",
           },
    'Coov':{"S+":"A1",
            "S-":"A2",
            "S-":"A2",
            "P":"E1",
            "D":"E2",
            "F":"E3",
           }
    }
    cs_trans = {
        "Dooh":"A1g",
        "Coov":"A1"
    }
    
    if mol.topgroup in sym_trans.keys():
        dct = sym_trans[mol.topgroup]
        new_syms = []
        for sym in syms:
            if sym == "CS":
                space_sym = cs_trans[mol.topgroup]
                new_syms += [f"1{space_sym}"]
            else:
                spin_sym = sym[0]
                space_sym = dct[sym[1:]] #translate
                new_syms += [spin_sym + space_sym]
        return new_syms
    else:
        return syms
    
def reduce_symmetries(mol,syms):
    """
    Here we need to reduce the listed literature symmetries to symmetries of their supported
    subgroups in the mcscf solver.
    
    I can't add every symmetry here -- if the point group of your molecule is not supported you
    will need to add it yourself
    
    For degenerate irreps, only one symmetry of the irrep is added to the mcscf solver.
    
    Additionally we translate 'CS' for "closed shell symmetry" to the totally symmetric irrep
    """
    reduce_pointgroup = {
        'C3v':'Cs',
        'D3h':'C2v',
        'D4h':'D2h',
        'D6h':'D2h',
        'Coov':'C2v',
        'Dooh':'D2h',
        'Oh':'D2h',
        'Td':'C2v',
        'SO3':'D2h',
    }
    sym_linear_trans = {
    'Dooh':{"Sg+":"A1g",
            "Sg-":"A2g",
            "Pg":"E1g",
            "Dg":"E2g",
            "Fg":"E3g",
            "Su+":"A1u",
            "Su-":"A2u",
            "Pu":"E1u",
            "Du":"E2u",
            "Fu":"E3u",
           },
    'Coov':{"S+":"A1",
            "S-":"A2",
            "S-":"A2",
            "P":"E1",
            "D":"E2",
            "F":"E3",
           }
    }
    sym_trans = {
        "C3v":{ #Goes to Cs
            "A1":["A'"],
            "A2":['A"'],
            "E":["A'",'A"'],
        },
        "D3h":{ #Goes to C2v
            "A1'":["A1"],
            "A2'":["B1"],
            'A1"':["A2"],
            'A2"':["B2"],
            "E'":["A1","B1"],
            'E"':["A2","B2"],
        },
        "D4h":{ #Goes to C2v
            "A1g":["Ag"],
            "A2g":["B1g"],
            "B1g":["Ag"],
            "B2g":["B1g"],
            "Eg":["B2g","B3g"],
            "A1u":["Au"],
            "A2u":["B1u"],
            "B1u":["Au"],
            "B2u":["B1u"],
            "Eu":["B2u","B3u"],
        },
        "D6h":{ #Goes to D2h
            "A1g":["Ag"],
            "A2g":["B1g"],
            "B1g":["B2g"],
            "B2g":["B3g"],
            "E1g":["B2g","B3g"],
            "E2g":["Ag","B1g"],
            "A1u":["Au"],
            "A2u":["B1u"],
            "B1u":["B3u"],
            "B2u":["B2u"],
            "E1u":["B2u","B3u"],
            "E2u":["B1u","Au"],
        },
        "Coov":{ #Goes to C2v
            "A1":["A1"],
            "A2":["A2"],
            "E1":["B1","B2"],
            "E2":["A1","A2"],
            "E3":["B1","B2"],
            "E4":["A1","A2"],
            "E5":["B1","B2"],
        },
        "Dooh":{ #Goes to D2h
            "A1g":["Ag"],
            "A2g":["B1g"],
            "A1u":["B1u"],
            "A2u":["Au"],
            "E1g":["B2g","B3g"],
            "E1u":["B2u","B3u"],
            "E2g":["Ag","B1g"],
            "E2u":["Au","B1u"],
            "E3g":["B2g","B3g"],
            "E3u":["B3u","B2u"],
            "E4g":["Ag","B1g"],
            "E4u":["Au","B1u"],
            "E5g":["B2g","B3g"],
            "E5u":["B3u","B2u"],
        },
        "Oh":{ #Goes to D2h
            "A1g":["Ag"],
            "A2g":["Ag"],
            "Eg":["Ag","Ag"],
            "T1g":["B1g","B2g","B3g"],
            "T2g":["B1g","B2g","B3g"],
            "A1u":["Au"],
            "A2u":["Au"],
            "Eu":["Au","Au"],
            "T1u":["B1u","B2u","B3u"],
            "T2u":["B1u","B2u","B3u"],
        },
        "Td":{ #Goes to C2v
            "A1":["A1"],
            "A2":["A2"],
            "E":["A1","A2"],
            "T1":["A2","B1","B2"],
            "T2":["A1","B1","B2"],
        },
        "SO3":{ #Goes to D2h
            "S":["Ag"],
            "P":["B1u","B2u","B3u"],
            "D":["B1g","B2g","B3g","Ag"],
        }
    }
    
    # target_irreps = {
    #     'Cs': ['A"', "A'"],
    #     'C2v': ['A1', 'A2', 'B1', 'B2'],
    #     'C2h': ['Ag', 'Bg', 'Au', 'Bu'],
    #     'D2h': ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
    # }

    if mol.topgroup in SUPPORTED_SYMS:
        return mol, syms

    new_syms = []
    for sym in syms:
        spin_sym = sym[0]
        space_sym = sym[1:]

        # Transform irrep notations of Coov and Dooh
        if mol.topgroup in sym_linear_trans.keys() and space_sym in sym_linear_trans[mol.topgroup]:
            space_sym = sym_linear_trans[mol.topgroup][space_sym]

        dct = sym_trans[mol.topgroup]

        if space_sym in dct.keys():
            space_sym = dct[space_sym][-1] #Translate -- take the last one for degenerate. Origin of ALL problems
            new_syms += [spin_sym + space_sym]
        # elif space_sym in target_irreps[mol.groupname]: #Already transformed
        #     new_syms += [spin_sym + space_sym]
        else:
            raise ValueError(f"Symmetry {sym} not found in {mol.topgroup} nor {mol.groupname}")
    return mol, new_syms
