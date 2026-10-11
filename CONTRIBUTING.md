# Genuine contributions by both members

The assignment requires exactly two members and meaningful Git commits from each. This generated starting point does not satisfy that human contribution requirement. No member names, IDs, emails or co-author trailers have been fabricated.

Each actual member should clone the repository, set their own Git identity, make changes they understand, run the relevant checks and commit their own work. Useful substantive contributions include explaining and improving the gradient implementation, reviewing the analysis against measured results, improving UI validation, adding an independently designed check, or conducting a separately labelled event-level experiment. A whitespace-only edit does not demonstrate meaningful technical contribution.

```bash
git clone https://github.com/laibasan/assignment-2.git
cd assignment-2
git switch -c your-change-name
# Review, understand, and make your own changes.
python -m unittest discover -s tests -v
git add <files-you-changed>
git commit -m "Explain the concrete change you made"
git push -u origin your-change-name
```

Use a pull request for collaboration. Keep the original test evaluation intact and document any new experiment separately. Review the assignment's AI prohibition with the instructor before using or submitting generated material.
