% =========================================================
% FEATURE SELECTION
% =========================================================

1 { circular; diamond; triangular; octagonal } 1.
1 { red_ground; white_ground; yellow_ground; blue } 1.

{ border }.
0 { black_border; red_border; white_border } 1.

{ bar }.
0 { black_bar; white_bar } 1.

{ symbol }.
0 { black_symbol; white_symbol } 1.


% =========================================================
% SYMBOL TYPES
% =========================================================

symbol_specific(symbol_stop).
symbol_specific(symbol_noentrygoods).
symbol_specific(symbol_overtaking).
symbol_specific(symbol_overtakinggoods).

symbol_specific(symbol_speed20).
symbol_specific(symbol_speed30).
symbol_specific(symbol_speed50).
symbol_specific(symbol_speed60).
symbol_specific(symbol_speed70).
symbol_specific(symbol_speed80).
symbol_specific(symbol_speed100).
symbol_specific(symbol_speed120).

symbol_specific(d1a1).
symbol_specific(d1a4).
symbol_specific(d1a5).
symbol_specific(d1a6).
symbol_specific(d1a7).
symbol_specific(d2a1).
symbol_specific(d2a2).
symbol_specific(d3).

% 0 { S : symbol_specific(S) } 1 :- symbol.
0 {
    symbol_stop;
    symbol_noentrygoods;
    symbol_overtaking;
    symbol_overtakinggoods;
    symbol_speed20;
    symbol_speed30;
    symbol_speed50;
    symbol_speed60;
    symbol_speed70;
    symbol_speed80;
    symbol_speed100;
    symbol_speed120;
    d1a1; d1a4; d1a5; d1a6; d1a7;
    d2a1; d2a2; d3
} 1 :- symbol.

%:- not symbol, symbol_specific(S), S.
:- not symbol, symbol_stop.
:- not symbol, symbol_noentrygoods.
:- not symbol, symbol_overtaking.
:- not symbol, symbol_overtakinggoods.
:- not symbol, symbol_speed20.
:- not symbol, symbol_speed30.
:- not symbol, symbol_speed50.
:- not symbol, symbol_speed60.
:- not symbol, symbol_speed70.
:- not symbol, symbol_speed80.
:- not symbol, symbol_speed100.
:- not symbol, symbol_speed120.
:- not symbol, d1a1.
:- not symbol, d1a4.
:- not symbol, d1a5.
:- not symbol, d1a6.
:- not symbol, d1a7.
:- not symbol, d2a1.
:- not symbol, d2a2.
:- not symbol, d3.


% =========================================================
% STRUCTURAL CONSTRAINTS
% =========================================================

:- border, not 1 { black_border; red_border; white_border } 1.
:- not border, black_border.
:- not border, red_border.
:- not border, white_border.

:- bar, not 1 { black_bar; white_bar } 1.
:- not bar, black_bar.
:- not bar, white_bar.

:- symbol, not 1 { black_symbol; white_symbol } 1.
:- not symbol, black_symbol.
:- not symbol, white_symbol.


% =========================================================
% HELPER RULES
% =========================================================

symbol_condition :- not symbol.
symbol_condition :- black_symbol.

bar_condition :- not bar.
bar_condition :- white_bar.

start_prohibition :-
    white_ground,
    symbol_condition,
    bar_condition.

start_prohibition :-
    yellow_ground,
    symbol_condition,
    bar_condition.

start_prohibition :-
    blue,
    red_border,
    symbol_condition,
    bar_condition.


% =========================================================
% CLASS RULES
% =========================================================

% Speed limits
class_1 :- circular, start_prohibition, symbol_speed20.
class_2 :- circular, start_prohibition, symbol_speed30.
class_3 :- circular, start_prohibition, symbol_speed50.
class_4 :- circular, start_prohibition, symbol_speed60.
class_5 :- circular, start_prohibition, symbol_speed70.
class_6 :- circular, start_prohibition, symbol_speed80.

class_7 :- circular, not border, white_ground, black_bar, symbol_speed80.
class_7 :- circular, not border, yellow_ground, black_bar, symbol_speed80.

class_8 :- circular, start_prohibition, symbol_speed100.
class_9 :- circular, start_prohibition, symbol_speed120.

% Prohibition
class_10 :- circular, start_prohibition, symbol_overtaking.
class_11 :- circular, start_prohibition, symbol_overtakinggoods.

% Danger
class_12 :- triangular, white_ground, red_border, symbol.
class_12 :- triangular, yellow_ground, red_border, symbol.

% Priority
class_13 :- diamond, yellow_ground, white_border.

class_14 :- triangular, white_ground, red_border, not symbol.
class_14 :- triangular, yellow_ground, red_border, not symbol.

class_15 :- octagonal, red_ground, white_symbol, symbol_stop.

% General prohibition
class_16 :- circular, white_ground, red_border, not symbol.
class_16 :- circular, yellow_ground, red_border, not symbol.

class_17 :- circular, start_prohibition, symbol_noentrygoods, not bar.

class_18 :- circular, red_ground, not symbol, white_bar.

% End prohibition
class_33 :- circular, white_ground, not border, black_bar, not symbol.
class_33 :- circular, yellow_ground, not border, black_bar, not symbol.

class_42 :- circular, white_ground, not border, black_bar, symbol_overtaking.
class_42 :- circular, yellow_ground, not border, black_bar, symbol_overtaking.

class_43 :- circular, white_ground, not border, black_bar, symbol_overtakinggoods.
class_43 :- circular, yellow_ground, not border, black_bar, symbol_overtakinggoods.

% Mandatory
d_base :- circular, blue, white_symbol.
d_base :- circular, white_ground, black_symbol.

class_34 :- d_base, d1a5.
class_35 :- d_base, d1a4.
class_36 :- d_base, d1a1.
class_37 :- d_base, d1a7.
class_38 :- d_base, d1a6.
class_39 :- d_base, d2a2.
class_40 :- d_base, d2a1.
class_41 :- d_base, d3.


% =========================================================
% EXACTLY ONE CLASS
% =========================================================

1 {
    class_1; class_2; class_3; class_4; class_5;
    class_6; class_7; class_8; class_9;
    class_10; class_11; class_12;
    class_13; class_14; class_15;
    class_16; class_17; class_18;
    class_33;
    class_34; class_35; class_36; class_37;
    class_38; class_39; class_40; class_41;
    class_42; class_43
} 1.


% =========================================================
% OUTPUT
% =========================================================

#show circular/0.
#show diamond/0.
#show triangular/0.
#show octagonal/0.

#show red_ground/0.
#show white_ground/0.
#show yellow_ground/0.
#show blue/0.

#show border/0.
#show black_border/0.
#show red_border/0.
#show white_border/0.

#show bar/0.
#show black_bar/0.
#show white_bar/0.

#show symbol/0.
#show black_symbol/0.
#show white_symbol/0.

#show symbol_stop/0.
#show symbol_noentrygoods/0.
#show symbol_overtaking/0.
#show symbol_overtakinggoods/0.
#show symbol_speed20/0.
#show symbol_speed30/0.
#show symbol_speed50/0.
#show symbol_speed60/0.
#show symbol_speed70/0.
#show symbol_speed80/0.
#show symbol_speed100/0.
#show symbol_speed120/0.

#show d1a1/0.
#show d1a4/0.
#show d1a5/0.
#show d1a6/0.
#show d1a7/0.
#show d2a1/0.
#show d2a2/0.
#show d3/0.

#show class_1/0.
#show class_2/0.
#show class_3/0.
#show class_4/0.
#show class_5/0.
#show class_6/0.
#show class_7/0.
#show class_8/0.
#show class_9/0.
#show class_10/0.
#show class_11/0.
#show class_12/0.
#show class_13/0.
#show class_14/0.
#show class_15/0.
#show class_16/0.
#show class_17/0.
#show class_18/0.
#show class_33/0.
#show class_34/0.
#show class_35/0.
#show class_36/0.
#show class_37/0.
#show class_38/0.
#show class_39/0.
#show class_40/0.
#show class_41/0.
#show class_42/0.
#show class_43/0.