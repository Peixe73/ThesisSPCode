from core.init import DO_SCRIPT_IMPORTS
from typing import TYPE_CHECKING
from collections import OrderedDict
from pathlib import Path
from functools import reduce
import operator

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets.binary_generator import BinaryGeneratorBuilder
    import torch
    from datetime import timedelta
    from core.util.progress_trackers import LogProgressContextManager
    import logging
    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=5))
    
# Auxiliary Function to enforce that a master concept is equivalent to the OR of its sub-concepts
# TODO: Not tested yet. Use with caution.
def conditional_one_hot(gen, master, vars):
    from functools import reduce
    import operator
    
    any_var = reduce(operator.or_, vars)

    # vars imply master
    for v in vars:
        gen.require(~v | master)

    # master implies at least one
    gen.require(~master | any_var)

    # mutual exclusion
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            gen.require(~(vars[i] & vars[j]))

def _build_ontology() -> 'BinaryGeneratorBuilder':
    gen = BinaryGeneratorBuilder()
    
    # Features
    Black_Bar = gen.free_variable()
    White_Bar = gen.free_variable()
    Bar = gen.implied_by(Black_Bar | White_Bar)
    
    Black_Border = gen.free_variable()
    Red_Border = gen.free_variable()
    White_Border = gen.free_variable()
    Border = gen.implied_by(Black_Border | Red_Border | White_Border)
    
    Red_Ground = gen.free_variable()
    White_Ground = gen.free_variable()
    Yellow_Ground = gen.free_variable()
    Blue_Ground = gen.free_variable()
    
    End_Prohibition = gen.free_variable()
    Start_Prohibition = gen.free_variable()
    
    Circular_Shape = gen.free_variable()
    Diamond_Shape = gen.free_variable()
    Triangular_Shape = gen.free_variable()
    Octagonal_Shape = gen.free_variable()
    
    Black_Symbol = gen.free_variable()
    White_Symbol = gen.free_variable()
    Symbol_NoEntryGoods = gen.free_variable()
    Symbol_Overtaking = gen.free_variable()
    Symbol_OvertakingGoods = gen.free_variable()
    Symbol_Speed20 = gen.free_variable()
    Symbol_Speed30 = gen.free_variable()
    Symbol_Speed50 = gen.free_variable()
    Symbol_Speed60 = gen.free_variable()
    Symbol_Speed70 = gen.free_variable()
    Symbol_Speed80 = gen.free_variable()
    Symbol_Speed100 = gen.free_variable()
    Symbol_Speed120 = gen.free_variable()
    Symbol_Stop = gen.free_variable()
    # Blue = gen.free_variable()
    D1a1 = gen.free_variable()
    D1a4 = gen.free_variable()
    D1a5 = gen.free_variable()
    D1a6 = gen.free_variable()
    D1a7 = gen.free_variable()
    D2a1 = gen.free_variable()
    D2a2 = gen.free_variable()
    D3a1 = gen.free_variable()
    
    """
    # Tirei o Blue
    Symbol = gen.implied_by(
        Symbol_NoEntryGoods | Symbol_Overtaking | Symbol_OvertakingGoods |
        Symbol_Speed20 | Symbol_Speed30 | Symbol_Speed50 | Symbol_Speed60 | Symbol_Speed70 | Symbol_Speed80 | Symbol_Speed100 | Symbol_Speed120 |
        Symbol_Stop | D1a1 | D1a4 | D1a5 | D1a6 | D1a7 | D2a1 | D2a2 | D3a1
    )
    """
    
    specific_symbols = [
        Symbol_NoEntryGoods, Symbol_Overtaking, Symbol_OvertakingGoods,
        Symbol_Speed20, Symbol_Speed30, Symbol_Speed50, Symbol_Speed60,
        Symbol_Speed70, Symbol_Speed80, Symbol_Speed100, Symbol_Speed120,
        Symbol_Stop, D1a1, D1a4, D1a5, D1a6, D1a7, D2a1, D2a2, D3a1
    ]
    
    Symbol = gen.free_variable()
    
    # If any specific symbol is true, then Symbol must be true
    for s in specific_symbols:
        gen.require(~s | Symbol)
    
    # If Symbol is true, then at least one specific symbol must be true
    any_symbol = reduce(operator.or_, specific_symbols)
    gen.require(~Symbol | any_symbol)
    
    # Ensure that no two specific symbols can be true at the same time
    for i in range(len(specific_symbols)):
        for j in range(i + 1, len(specific_symbols)):
            gen.require(~(specific_symbols[i] & specific_symbols[j]))
            
    #Same Thing but for colors
    color_symbols = [Black_Symbol, White_Symbol]

    conditional_one_hot(gen, Symbol, color_symbols)
    
    gen.features = OrderedDict(
        Black_Bar=Black_Bar,
        White_Bar=White_Bar,
        Bar=Bar,
        
        Black_Border=Black_Border,
        Red_Border=Red_Border,
        White_Border=White_Border,
        Border=Border,
        
        Red_Ground=Red_Ground,
        White_Ground=White_Ground,
        Yellow_Ground=Yellow_Ground,
        Blue_Ground=Blue_Ground,
        
        End_Prohibition=End_Prohibition,
        Start_Prohibition=Start_Prohibition,
        
        Circular_Shape=Circular_Shape,
        Diamond_Shape=Diamond_Shape,
        Triangular_Shape=Triangular_Shape,
        Octagonal_Shape=Octagonal_Shape,
        
        Symbol=Symbol,
        
        Black_Symbol=Black_Symbol,
        White_Symbol=White_Symbol,

        Symbol_NoEntryGoods=Symbol_NoEntryGoods,
        Symbol_Overtaking=Symbol_Overtaking,
        Symbol_OvertakingGoods=Symbol_OvertakingGoods,

        Symbol_Speed20=Symbol_Speed20,
        Symbol_Speed30=Symbol_Speed30,
        Symbol_Speed50=Symbol_Speed50,
        Symbol_Speed60=Symbol_Speed60,
        Symbol_Speed70=Symbol_Speed70,
        Symbol_Speed80=Symbol_Speed80,
        Symbol_Speed100=Symbol_Speed100,
        Symbol_Speed120=Symbol_Speed120,

        Symbol_Stop=Symbol_Stop,

        D1a1=D1a1,
        D1a4=D1a4,
        D1a5=D1a5,
        D1a6=D1a6,
        D1a7=D1a7,
        D2a1=D2a1,
        D2a2=D2a2,
        D3a1=D3a1
    )    
    
    # Intermediary Concepts
    
    WhiteOrYellow_Ground = White_Ground | Yellow_Ground
    
    WhiteOrYellowOrBlue_Ground = WhiteOrYellow_Ground | Blue_Ground
    
    # Danger
    Danger_Sign = Triangular_Shape & WhiteOrYellow_Ground & Red_Border & Symbol
    # A1a_Sign = Danger_Sign & D1a1
    
    #Priority
    B1_Sign = Triangular_Shape & WhiteOrYellow_Ground & Red_Border & ~ Symbol
    B2_Sign = Octagonal_Shape & Red_Ground & White_Border & Symbol_Stop & White_Symbol
    B3_Sign = Diamond_Shape & Yellow_Ground & White_Border
    Priority_Sign = B1_Sign | B2_Sign | B3_Sign
    
    #Prohibitory or Restrictive
    C_Sign_Shape = Circular_Shape
    # Falta Red Symbol e Red Bar, que tb não está na ontologia :)
    Start_Prohibition = WhiteOrYellowOrBlue_Ground & Red_Border & Black_Symbol & White_Bar
    C1a_Sign = C_Sign_Shape & Red_Ground & ~ Symbol & White_Bar
    C2_Sign = C_Sign_Shape & WhiteOrYellow_Ground & Red_Border & ~ Symbol
    C3e3_Sign = C_Sign_Shape & Start_Prohibition & Symbol_NoEntryGoods & ~ Bar
    C13aa_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Overtaking & ~ Bar
    C13bb_Sign = C_Sign_Shape & Start_Prohibition & Symbol_OvertakingGoods & ~ Bar
    C14_20_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed20
    C14_30_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed30
    C14_50_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed50
    C14_60_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed60
    C14_70_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed70
    C14_80_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed80
    C14_100_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed100
    C14_120_Sign = C_Sign_Shape & Start_Prohibition & Symbol_Speed120
    C17a_Sign = C_Sign_Shape & WhiteOrYellow_Ground & ~ Border & Black_Bar & ~ Symbol
    C17b_80_Sign = C_Sign_Shape & WhiteOrYellow_Ground & ~ Border & Black_Bar & Symbol_Speed80
    C17c_Sign = C_Sign_Shape & WhiteOrYellow_Ground & ~ Border & Black_Bar & Symbol_Overtaking
    C17d_Sign = C_Sign_Shape & WhiteOrYellow_Ground & ~ Border & Black_Bar & Symbol_OvertakingGoods
    
    Prohibitive_Sign = C1a_Sign | C2_Sign | C3e3_Sign | C13aa_Sign | C13bb_Sign | C14_20_Sign | C14_30_Sign | C14_50_Sign | C14_60_Sign | C14_70_Sign | C14_80_Sign | C14_100_Sign | C14_120_Sign | C17a_Sign | C17b_80_Sign | C17c_Sign | C17d_Sign
    
    # Mandatory
    # Falta Rectangular_Shape, que tb não está na ontologia :)
    D_Sign_Shape = Circular_Shape
    BlueGroundAndWhiteSymbol = Blue_Ground & White_Symbol
    WhiteGroundAndBlackSymbol = White_Ground & Black_Symbol
    Mandatory_Sign_Partial = BlueGroundAndWhiteSymbol | WhiteGroundAndBlackSymbol
    Mandatory_Sign = D_Sign_Shape & Mandatory_Sign_Partial
    D1a1_Sign = Mandatory_Sign & D1a1
    D1a4_Sign = Mandatory_Sign & D1a4
    D1a5_Sign = Mandatory_Sign & D1a5
    D1a6_Sign = Mandatory_Sign & D1a6
    D1a7_Sign = Mandatory_Sign & D1a7
    D2a1_Sign = Mandatory_Sign & D2a1
    D2a2_Sign = Mandatory_Sign & D2a2
    D3a1_Sign = Mandatory_Sign & D3a1
    
    gen.labels.update(
        Danger_Sign = Danger_Sign,
        Priority_Sign = Priority_Sign,
        Prohibitive_Sign = Prohibitive_Sign,
        Mandatory_Sign = Mandatory_Sign,
        Start_Prohibition = Start_Prohibition,
        C1a_Sign = C1a_Sign,
        C2_Sign = C2_Sign,
        C3e3_Sign = C3e3_Sign,
        C13aa_Sign = C13aa_Sign,
        C13bb_Sign = C13bb_Sign,
        C14_20_Sign = C14_20_Sign,
        C14_30_Sign = C14_30_Sign,
        C14_50_Sign = C14_50_Sign,
        C14_60_Sign = C14_60_Sign,
        C14_70_Sign = C14_70_Sign,
        C14_80_Sign = C14_80_Sign,
        C14_100_Sign = C14_100_Sign,
        C14_120_Sign = C14_120_Sign,
        C17a_Sign = C17a_Sign,
        C17b_80_Sign = C17b_80_Sign,
        C17c_Sign = C17c_Sign,
        C17d_Sign = C17d_Sign,
        D1a1_Sign = D1a1_Sign,
        D1a4_Sign = D1a4_Sign,
        D1a5_Sign = D1a5_Sign,
        D1a6_Sign = D1a6_Sign,
        D1a7_Sign = D1a7_Sign,
        D2a1_Sign = D2a1_Sign,
        D2a2_Sign = D2a2_Sign,
        D3a1_Sign = D3a1_Sign
    )
    
    return gen

PATH = Path("data/gtsrb_ontology.csv")

def main():
    generator = _build_ontology().build()
    feature_names = generator.feature_names
    label_names = generator.label_names
    assert feature_names is not None and label_names is not None
    header = feature_names + label_names + [generator.valid_label]
    if PATH.exists():
        raise FileExistsError(f"{PATH} already exists")
    with open(PATH, "w") as f:
        f.write(",".join(header) + "\n")
        with progress_cm.track("Dataset Generation", "rows", len(generator)) as progress:
            for i in range(len(generator)):
                row = generator.generate_from_int(i, force_valid=True)
                row = torch.cat(row)
                row = row.tolist()
                f.write(",".join(map(str, row)) + "\n")
                progress.tick()
    
    
    
    
    
    
    #Post = gen.free_variable()

    """
    # Features
    two_passenger = gen.free_variable()
    two_freight = gen.free_variable()
    long_passenger = gen.free_variable()
    two_long_wagon = gen.free_variable() # not a feature
    three_wagon = gen.free_variable()

    passenger_car = gen.implied_by(two_passenger | long_passenger)
    freight_wagon = gen.implied_by(two_freight)
    empty_wagon = gen.implied_by(
        (three_wagon | two_long_wagon) & # then it must have at least 1 (unspecified) wagon
        # default to empty if no other type is specified
        ~ passenger_car &
        ~ freight_wagon
    )

    long_wagon = gen.implied_by(two_long_wagon | long_passenger)

    reinforced_car = gen.free_variable()

    # store all nodes so far as features
    gen.features = OrderedDict(
        PassengerCar=passenger_car,
        FreightWagon=freight_wagon,
        EmptyWagon=empty_wagon,
        LongWagon=long_wagon,
        ReinforcedCar=reinforced_car,
        LongPassengerCar=long_passenger,
        AtLeast2PassengerCars=two_passenger,
        AtLeast2FreightWagons=two_freight,
        AtLeast3Wagons=three_wagon,
        AtLeast2LongWagons=two_long_wagon
    )

    # Intermediary concepts (added for readability)
    all_empty = ~ passenger_car & ~ freight_wagon
    passenger_car_or_freight_wagon = passenger_car | freight_wagon

    # Intermediary concepts (present in the original ontology)
    empty_train = all_empty & empty_wagon
    long_train = two_long_wagon | three_wagon

    # Simplified intermediary concepts:
    #       the following equivalences were reverse implications in the original ontology
    war_train = reinforced_car & passenger_car
    passenger_train = long_passenger | two_passenger
    freigh_train = two_freight
    rural_train = empty_wagon & passenger_car_or_freight_wagon & ~ long_wagon
    mixed_train = passenger_car & freight_wagon & empty_wagon

    # More intermediary concepts (present in the original ontology)
    # These were always equivalences.
    long_freight_train = long_train & freigh_train

    # Labels
    type_a = war_train | empty_train
    type_b = passenger_train | long_freight_train
    type_c = rural_train | mixed_train
    other = ~ (type_a | type_b | type_c)
    gen.labels.update(
        TypeA = type_a,
        TypeB = type_b,
        TypeC = type_c,
        Other = other,
        WarTrain = war_train,
        PassengerTrain = passenger_train,
        FreightTrain = freigh_train,
        RuralTrain = rural_train,
        MixedTrain = mixed_train,
        LongTrain = long_train,
        EmptyTrain = empty_train,
        LongFreightTrain = long_freight_train
    )

    return gen

PATH = Path("data/xtrains_ontology.csv")

def main():
    generator = _build_ontology().build()
    feature_names = generator.feature_names
    label_names = generator.label_names
    assert feature_names is not None and label_names is not None
    header = feature_names + label_names + [generator.valid_label]
    if PATH.exists():
        raise FileExistsError(f"{PATH} already exists")
    with open(PATH, "w") as f:
        f.write(",".join(header) + "\n")
        with progress_cm.track("Dataset Generation", "rows", len(generator)) as progress:
            for i in range(len(generator)):
                row = generator.generate_from_int(i, force_valid=False)
                row = torch.cat(row)
                row = row.tolist()
                f.write(",".join(map(str, row)) + "\n")
                progress.tick()
"""
