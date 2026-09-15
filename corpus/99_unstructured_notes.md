# Miscellaneous Notes (Sample Corpus)

**Document ID:** 99_unstructured_notes.md

This document intentionally does not follow either of the two known
markdown formats used elsewhere in this corpus. It has no "Reference
Sentences" section and no "Key Points" section, so the ingestion
pipeline should route it through the fixed-size fallback chunker instead
of failing or silently dropping it.

It exists purely to exercise that fallback path during testing. In a
real corpus this might be a loose collection of scratch notes, a draft,
or a document that hasn't been reformatted into the standard structure
yet. The pipeline should still index it -- just with lower-quality,
non-semantic chunk boundaries -- and the ingestion report should flag it
by name so a human can go fix its structure later if desired.

Repeating filler content below purely to give the fallback chunker
enough words to produce more than one fixed-size window during testing,
so the overlap behavior between windows can also be verified end to end
without needing a much larger fixture file for this one purpose alone.
Repeating filler content below purely to give the fallback chunker
enough words to produce more than one fixed-size window during testing,
so the overlap behavior between windows can also be verified end to end
without needing a much larger fixture file for this one purpose alone.
Repeating filler content below purely to give the fallback chunker
enough words to produce more than one fixed-size window during testing,
so the overlap behavior between windows can also be verified end to end
without needing a much larger fixture file for this one purpose alone.
