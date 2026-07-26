# M6F Human Review Handoff

M6F is locally `PASS` with independent `P0=0/P1=0/P2=0`.

The repaired Broker now binds the complete request, proposal-generation
session, logical call, and content-bound release across both stored-row and
missing-row recovery. Six exact replays preserved their original evidence;
24 conflicting replays failed before response reading or process spawn.

The content-bound Broker release is
`2d88e3df3486f2369c4487c66c333dbf046796c8d38e449e506c184c7f9fadd1`.
Both treatment-free real conformance shapes passed under that release.

Pilot V5 remains sealed and non-reusable. Its provider-level failure cause
remains `UNKNOWN`.

Exactly one fresh Pilot is authorized, but its contract must bind the completed
parallel Meta checkpoint rather than freezing the older Meta source while that
work is still in progress. Until that identity is available, no Pilot root,
request, SearchRound, Main freeze, M7, or M8 should be opened.

Authority remains `NONE`; evidence remains `DEVELOPMENT_ONLY`; formal
acceptance remains false. Nothing was pushed.
