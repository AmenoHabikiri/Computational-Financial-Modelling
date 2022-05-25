/*
	Name - Sagar Tarafdar
	Roll Number - B19110
	Course - Paradigms of Programming (Lab 6)
*/

% Question 1
myLength([], 0).
myLength([_|Xs], L) :- myLength(Xs, N) , L is N+1. 

% Question 2
myLast(X, [X]).
myLast(X, [_|Xs]) :- myLast(X, Xs).

% Question 3
% Bread = butter. Unifiable , Binding -> Bread = Butter
% bread = butter. Not Unifiable, Both are constants
% food(bread,X) = food(Y,butter). Unifiable, Binding -> X = butter, Y = bread
% food(bread,X,butter) = food(Y,cheese,X). Not Unifiable 

% Question 4
rooms([room(_,4),room(_,3),room(_,2),room(_,1)]).
hostel(Rooms) :- rooms(Rooms),
 member(room(sagar, A), Rooms), A \= 2,
 member(room(shashwat, D), Rooms),
 member(room(subham, B), Rooms),
 member(room(saloni, C), Rooms), C \= 1,
 not(adjacent(B, C)), adjacent(B, D), adjacent(A, C), B < A,
 print_rooms(Rooms).

adjacent(X, Y) :- X =:= Y+1.
adjacent(X, Y) :- X =:= Y-1.
print_rooms([A | B]) :- write(A), nl, print_rooms(B).
print_rooms([]).

% Possible Solutions
% Rooms = [room(sagar, 4), room(saloni, 3), room(shashwat, 2), room(subham, 1)] ;
% Rooms = [room(sagar, 4), room(saloni, 3), room(shashwat, 2), room(subham, 1)] ;
% Rooms = [room(saloni, 4), room(sagar, 3), room(subham, 2), room(shashwat, 1)] ;
