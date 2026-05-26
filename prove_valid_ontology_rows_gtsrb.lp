1 { circular; diamond; triangular; octagonal } 1.
1 { red_ground; white_ground; yellow_ground; blue } 1.

{ border }.
%{black_border}.
%{red_border}.
%{white_border}.
0 { black_border; red_border; white_border } 1.

{ bar }.
%{ black_bar }.
%{ white_bar }.
0 { black_bar; white_bar } 1.

{ symbol }.
%{ black_symbol }.
%{ white_symbol }.
%{symbol_stop}.
%{symbol_noentrygoods}.
%{symbol_overtaking}.
%{symbol_overtakinggoods}.
%{symbol_speed20}.
%{symbol_speed30}.
%{symbol_speed50}.
%{symbol_speed60}.
%{symbol_speed70}.
%{symbol_speed80}.
%{symbol_speed100}.
%{symbol_speed120}.
%{ d1a1; d1a4; d1a5; d1a6; d1a7 }.
%{ d2a1; d2a2 }.
%{ d3 }.
0 { black_symbol; white_symbol } 1.


% Symbols

warning_symbol_exception :-
    triangular,
    red_border,
    white_ground.

warning_symbol_exception :-
    triangular,
    red_border,
    yellow_ground.

1 {
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
} 1 :- symbol,
       not warning_symbol_exception.


% If no symbol is present, none of the symbol atoms can appear
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


% Structural constraints

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


% Auxiliary logic

symbol_condition :- black_symbol.
symbol_condition :- not symbol.

bar_condition :- not bar.
bar_condition :- white_bar.

start_prohibition :-
    white_ground, symbol_condition, bar_condition.%black_symbol, not bar.%white_bar.%bar_condition. %symbol_condition

start_prohibition :-
    yellow_ground, symbol_condition, bar_condition.%black_symbol, not bar.%white_bar.%bar_condition. %symbol_condition

start_prohibition :-
    blue, red_border, symbol_condition, bar_condition.%black_symbol, not bar.%white_bar.%bar_condition. %symbol_condition


% Valid Class Definitions

% ---- Speed ----
valid(1) :- circular, start_prohibition, symbol_speed20.
valid(2) :- circular, start_prohibition, symbol_speed30.
valid(3) :- circular, start_prohibition, symbol_speed50.
valid(4) :- circular, start_prohibition, symbol_speed60. 
valid(5) :- circular, start_prohibition, symbol_speed70.
valid(6) :- circular, start_prohibition, symbol_speed80.

valid(7) :- circular, not border, white_ground, black_bar, symbol_speed80.
valid(7) :- circular, not border, yellow_ground, black_bar, symbol_speed80.

valid(8) :- circular, start_prohibition, symbol_speed100.
valid(9) :- circular, start_prohibition, symbol_speed120.

% ---- Prohibition ----
valid(10) :- circular, start_prohibition, symbol_overtaking, not bar.
valid(11) :- circular, start_prohibition, symbol_overtakinggoods, not bar.

% ---- Warning ----
valid(12) :- triangular, symbol, red_border, white_ground, not bar, black_symbol.
valid(12) :- triangular, symbol, red_border, yellow_ground, not bar, black_symbol.
%ground_ok :- white_ground.
%ground_ok :- yellow_ground.

%valid(12) :-
%    triangular,
%    red_border,
%    ground_ok,
%    symbol,
%    black_symbol,
%    not bar,
%    1 { symbol_speed20; symbol_speed30; symbol_speed50;
%        symbol_speed60; symbol_speed70; symbol_speed80;
%        symbol_speed100; symbol_speed120;
%        symbol_overtaking; symbol_overtakinggoods;
%        symbol_noentrygoods; symbol_stop;
%        d1a1; d1a4; d1a5; d1a6; d1a7;
%        d2a1; d2a2; d3 } 1.

% ---- Priority ----
valid(13) :- diamond, yellow_ground, white_border, not symbol, not bar.

valid(14) :- triangular, white_ground, red_border, not symbol, not bar.
valid(14) :- triangular, yellow_ground, red_border, not symbol, not bar.

valid(15) :- octagonal, red_ground, white_symbol, symbol_stop, not bar.

% ---- General prohibition ----
valid(16) :- circular, white_ground, red_border, not symbol, not bar.
valid(16) :- circular, yellow_ground, red_border, not symbol, not bar.

valid(17) :- circular, start_prohibition, symbol_noentrygoods, not bar.
valid(18) :- circular, red_ground, not symbol, white_bar.

% ---- End prohibition ----
valid(33) :- circular, white_ground, not border, black_bar, not symbol.
valid(33) :- circular, yellow_ground, not border, black_bar, not symbol.

valid(42) :- circular, white_ground, not border, black_bar, symbol_overtaking, black_symbol.
valid(42) :- circular, yellow_ground, not border, black_bar, symbol_overtaking, black_symbol.

valid(43) :- circular, white_ground, not border, black_bar, symbol_overtakinggoods, black_symbol.
valid(43) :- circular, yellow_ground, not border, black_bar, symbol_overtakinggoods, black_symbol.

% ---- Mandatory / Directional ----
d_base :- circular, blue, white_symbol, not border.
d_base :- circular, white_ground, black_symbol, not border.

valid(34) :- d_base, d1a5.
valid(35) :- d_base, d1a4.
valid(36) :- d_base, d1a1.
valid(37) :- d_base, d1a7.
valid(38) :- d_base, d1a6.
valid(39) :- d_base, d2a2.
valid(40) :- d_base, d2a1.
valid(41) :- d_base, d3.


% Exactly only one class

1 { class(1..43) } 1.

:- class(C), not valid(C).


% Outputs

#show class/1.
#show valid/1.
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