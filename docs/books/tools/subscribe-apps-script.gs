/**
 * Squish Subscribers web hook (Google Apps Script).
 *
 * Receives sign-ups from squish.run/books, appends a row to the
 * "Squish Subscribers" Google Sheet and emails Wes on every new sign-up.
 *
 * Setup (once):
 *   1. Open the "Squish Subscribers" sheet > Extensions > Apps Script.
 *   2. Replace the editor contents with this file and click Save.
 *   3. Deploy > New deployment > type "Web app".
 *      Execute as: Me.  Who has access: Anyone.  Click Deploy and allow access.
 *   4. Copy the Web app URL (ends in /exec) and paste it into
 *      SHEET_ENDPOINT in docs/books/index.html.
 * After editing this script later: Deploy > Manage deployments > Edit > New version.
 */

var NOTIFY_EMAIL = "wesleyscholl@gmail.com";
var SHEET_NAME = ""; // empty = first sheet tab
var EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function doPost(e) {
  try {
    var data = JSON.parse((e && e.postData && e.postData.contents) || "{}");

    // Honeypot: real people never fill the hidden "website" field.
    if (data.website) {
      return reply({ ok: true });
    }

    var email = String(data.email || "").trim().toLowerCase().slice(0, 200);
    var firstName = String(data.firstName || "").trim().slice(0, 80);
    var source = String(data.src || "(none)").trim().slice(0, 80);
    var page = String(data.page || "").trim().slice(0, 200);
    var consent = data.consent === true || data.consent === "true";

    if (!EMAIL_PATTERN.test(email)) {
      return reply({ ok: false, error: "invalid_email" });
    }
    if (!consent) {
      return reply({ ok: false, error: "no_consent" });
    }

    var lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      var ss = SpreadsheetApp.getActiveSpreadsheet();
      var sheet = SHEET_NAME ? ss.getSheetByName(SHEET_NAME) : ss.getSheets()[0];
      var emails = sheet.getRange(1, 3, Math.max(sheet.getLastRow(), 1), 1).getValues();
      var already = emails.some(function (row) {
        return String(row[0]).trim().toLowerCase() === email;
      });
      if (!already) {
        sheet.appendRow([new Date(), firstName, email, source, page, "Yes (web form)"]);
      }
      MailApp.sendEmail({
        to: NOTIFY_EMAIL,
        subject: already ? "Squish sign-up (already on list): " + email : "New Squish subscriber: " + (firstName || email),
        body:
          "Name: " + (firstName || "(not given)") + "\n" +
          "Email: " + email + "\n" +
          "Source: " + source + "\n" +
          "Page: " + page + "\n" +
          "Time: " + new Date() + "\n\n" +
          "Sheet: " + SpreadsheetApp.getActiveSpreadsheet().getUrl()
      });
    } finally {
      lock.releaseLock();
    }
    return reply({ ok: true });
  } catch (err) {
    console.error("Squish subscribe failed: " + err);
    return reply({ ok: false, error: "server_error" });
  }
}

function doGet() {
  return reply({ ok: true, service: "squish-subscribers" });
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
