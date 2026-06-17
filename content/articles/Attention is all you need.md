+++
date = '2026-04-09T10:28:55+05:30'
draft = true
title = 'Annotated Attention Is All You Need'
+++

### Started on 10th of March 2026

[More annotated transformers.](https://pi-tau.github.io/posts/transformer/)

Before understanding this architecture I wanted to understand what came before this
and what were the problems that previous ones couldn't solve - were solved by this
architecture we call transformers.

What is the meaning or purpose behind this architecture - what problem(s)
exactly does it solves? Or why this exists at all?

So lets start with the most basic of problem - how do we get machines or
computers to understand human language?
The understanding of numbers, characters, alphabets was done a long before
by reducing that data into bits. Lets consider the example of numbers first.
How do humans recognize numbers? If I were to show to you 3 different images
of each drawing number 8, although the exact pattern of strokes and edges might
differ in each image but somehow your brain still recognize this as the same idea
being represented in rest of the 2 images.

But the mechanism of how computer interpret numbers is a whole different story by the way.
Since numbers(also other data types) are reduced to bits in order for the computer to
"understand" for what it actually is. This is **NOT** learning, this is just interpretation.
A normal computer understands numbers using explicit encoding rules that were predefined
by some human. What we're trying to achieve here is to build a program/algorithm that
could actually understand or learn this representation on its own without the need of
any explicit rules.
