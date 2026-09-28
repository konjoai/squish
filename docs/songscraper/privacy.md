# Privacy Policy — songscraper

_Last updated: September 27, 2026_

songscraper is a personal, single-user tool built and operated by Wesley Scholl
(wesleyscholl@gmail.com). It is not offered to the public and has no other users.

## What data it accesses

When the operator (the sole authorized user) triggers a scrape, songscraper uses that Google
account's consent to:

- **Read Drive file metadata** (`drive.metadata.readonly`) — to locate the Google Doc template.
- **Create and edit Google Docs** (`documents`, `drive.file`) — to copy the template and fill it
  in with the scraped chord chart (title, chords, lyrics).
- **Access Google Drive** (`drive`) — to copy the template file and place the finished document
  in the configured Drive folder.

songscraper does not read, modify, or delete any other files in Drive, and does not access
Gmail, Calendar, Contacts, or any other Google service.

## How the data is used

Google user data is used **solely** to create the chord-chart Google Doc requested for that one
scrape. It is never used for advertising, profiling, model training, or any purpose beyond
producing that document.

## Storage and retention

songscraper is a stateless service — it does not operate a database and retains no Google user
data between requests. The only persistent result of a scrape is the Google Doc itself, created
directly in the operator's own Google Drive under their own Drive's normal access controls;
songscraper does not keep a separate copy anywhere.

## Sharing

songscraper does not share, sell, or transfer any Google user data to any third party, for any
purpose.

songscraper's use and transfer of information received from Google APIs to any other app will
adhere to the [Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including the Limited Use requirements.

## Revoking access

Access can be revoked at any time at
[myaccount.google.com/permissions](https://myaccount.google.com/permissions).

## Contact

Questions about this policy or this data: wesleyscholl@gmail.com
