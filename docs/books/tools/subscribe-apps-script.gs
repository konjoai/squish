/**
 * Squish Subscribers web hook (Google Apps Script).
 *
 * Receives sign-ups from squish.run/books (the Join the Squish List page and
 * the free printables form), keeps the "Squish Subscribers" Google Sheet up to
 * date, sends each new person a welcome email with the free coloring pages,
 * gives every email a working unsubscribe link, and emails Wes on every sign-up.
 *
 * Setup (once):
 *   1. Open the "Squish Subscribers" sheet > Extensions > Apps Script.
 *   2. Replace the editor contents with this file and click Save.
 *   3. Fill in POSTAL_ADDRESS below (a PO box or mailbox address is fine).
 *      US law (CAN-SPAM) requires a postal address in marketing email. Until it
 *      is filled in, no welcome email is sent; Wes gets a note to send the
 *      coloring pages by hand instead.
 *   4. Deploy > Manage deployments > Edit (pencil) > Version: New version > Deploy.
 *      Keep "Execute as: Me" and "Who has access: Anyone". The /exec URL stays
 *      the same, so the website does not need to change.
 *   5. Run testSetup() once from the editor and approve the permissions prompt.
 *
 * Sheet columns: Timestamp | First name | Email | Source | Page | Consent | Status | Welcome sent
 * Status values: Subscribed, Printables only, Unsubscribed.
 *
 * Limits: a personal Gmail account can send email to about 100 recipients a day
 * from Apps Script, and each sign-up uses 2 (welcome + note to Wes), so about
 * 45 sign-ups a day before welcomes pause. The script checks the remaining quota before every welcome
 * email. Past roughly 80 to 100 subscribers, move monthly emails to a free email
 * service and keep this sheet as the backup.
 */

var NOTIFY_EMAIL = "wesleyscholl@gmail.com";
var SENDER_NAME = "Wes from Learning with Squish";
var POSTAL_ADDRESS = ""; // e.g. "Konjo Publishing, PO Box 0000, Alexandria, VA 22300"
var SITE = "https://squish.run/books/";
// The deployed web app URL (same one as SHEET_ENDPOINT on the website). Used for
// unsubscribe links, because ScriptApp.getService().getUrl() can return a /dev
// link or null depending on how the script is run.
var EXEC_URL = "https://script.google.com/macros/s/AKfycbxqoJqZ8XiZ5y785j-FrOdQO3_JOtBi8zBR8ZiPrtskLzzwtipWLAi57LU75yA5hLvG9A/exec";
var PRINTABLES_URL = SITE + "downloads/squish-lilyan-coloring-pages.pdf";
var SHEET_NAME = ""; // empty = first sheet tab
var EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
var HEADERS = ["Timestamp", "First name", "Email", "Source", "Page", "Consent", "Status", "Welcome sent"];
var COL = { email: 3, consent: 6, status: 7, welcome: 8 };

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
    var purpose = String(data.purpose || "list").trim();

    if (!EMAIL_PATTERN.test(email)) {
      return reply({ ok: false, error: "invalid_email" });
    }
    // The Join page requires consent. The printables form may send without it
    // (the person only asked for the coloring pages).
    if (!consent && purpose !== "printables") {
      return reply({ ok: false, error: "no_consent" });
    }

    var result;
    var lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      result = upsertRow(email, firstName, source, page, consent);
    } finally {
      lock.releaseLock();
    }

    if (result.sendWelcome) {
      var sent = sendWelcome(email, firstName, result.status);
      if (sent) {
        getSheet().getRange(result.row, COL.welcome).setValue(new Date());
      }
    }
    try {
      notifyWes(email, firstName, source, page, result);
    } catch (notifyErr) {
      // Out of daily email quota: the row is saved, so don't fail the visitor.
      console.warn("Squish notify skipped: " + notifyErr);
    }
    return reply({ ok: true });
  } catch (err) {
    console.error("Squish subscribe failed: " + err);
    return reply({ ok: false, error: "server_error" });
  }
}

/** Unsubscribe link target: ?u=<email>&t=<token>. Also answers health checks. */
function doGet(e) {
  var params = (e && e.parameter) || {};
  if (!params.u) {
    return reply({ ok: true, service: "squish-subscribers" });
  }
  var email = String(params.u).trim().toLowerCase();
  if (params.t !== tokenFor(email)) {
    return htmlPage("That unsubscribe link didn't work. Reply to any Squish email and Wes will remove you by hand.");
  }
  // Two steps, so email link scanners that open links can't unsubscribe anyone.
  if (params.confirm !== "1") {
    var yes = unsubscribeUrl(email) + "&confirm=1";
    return htmlPage('Unsubscribe <b>' + email.replace(/[<>&"]/g, "") + '</b> from the Squish List?<br><br>' +
      '<a href="' + yes + '" target="_top" style="display:inline-block;background:#FFD34D;color:#3B1F5C;padding:10px 18px;border-radius:12px;font-weight:bold;text-decoration:none">Yes, unsubscribe me</a>');
  }
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    var sheet = getSheet();
    var row = findRow(sheet, email);
    if (row) {
      sheet.getRange(row, COL.status).setValue("Unsubscribed");
    }
  } finally {
    lock.releaseLock();
  }
  MailApp.sendEmail(NOTIFY_EMAIL, "Squish List unsubscribe: " + email, "Marked Unsubscribed in the sheet.");
  return htmlPage("You're unsubscribed from the Squish List. Thanks for reading with Squish!");
}

function upsertRow(email, firstName, source, page, consent) {
  var sheet = getSheet();
  ensureHeaders(sheet);
  var status = consent ? "Subscribed" : "Printables only";
  var row = findRow(sheet, email);
  if (!row) {
    sheet.appendRow([new Date(), firstName, email, source, page,
      consent ? "Yes (web form)" : "No (printables only)", status, ""]);
    return { row: sheet.getLastRow(), isNew: true, status: status, sendWelcome: true };
  }
  var current = String(sheet.getRange(row, COL.status).getValue() || "Subscribed");
  // A new opt-in upgrades a printables-only or unsubscribed row; it never downgrades.
  if (consent && current !== "Subscribed") {
    sheet.getRange(row, COL.consent).setValue("Yes (web form)");
    sheet.getRange(row, COL.status).setValue("Subscribed");
    current = "Subscribed";
  }
  var welcomed = sheet.getRange(row, COL.welcome).getValue();
  return { row: row, isNew: false, status: current, sendWelcome: !welcomed && current !== "Unsubscribed" };
}

function sendWelcome(email, firstName, status) {
  if (!POSTAL_ADDRESS) {
    MailApp.sendEmail(NOTIFY_EMAIL, "Squish List: welcome email not sent (add POSTAL_ADDRESS)",
      "Add a postal address to the script so welcome emails can go out. Meanwhile send the coloring pages to " +
      email + " by hand: " + PRINTABLES_URL);
    return false;
  }
  // Keep a margin: each sign-up also uses one recipient for the note to Wes.
  if (MailApp.getRemainingDailyQuota() < 10) {
    MailApp.sendEmail(NOTIFY_EMAIL, "Squish List: welcome email skipped (daily quota low)",
      "Send the coloring pages to " + email + " by hand: " + PRINTABLES_URL);
    return false;
  }
  var hello = firstName ? "Hi " + firstName + "," : "Hi there,";
  var lines = [
    hello,
    "",
    "Thanks for stopping by Learning with Squish! Here are your free Squish and Lilyan coloring pages:",
    PRINTABLES_URL,
    "",
    status === "Subscribed"
      ? "You're on the Squish List, so I'll email you about once a month with new books, more free printables and local storytimes."
      : "That's the only email you'll get from me unless you join the Squish List: " + SITE + "subscribe/",
    "",
    "All the books are here: " + SITE,
    "",
    "Happy coloring,",
    "Wes (and Lilyan and Squish)",
    "",
    "--",
    "Learning with Squish, Konjo Publishing",
    POSTAL_ADDRESS,
    "Unsubscribe anytime: " + unsubscribeUrl(email)
  ];
  MailApp.sendEmail({
    to: email,
    subject: "Your free Squish coloring pages",
    body: lines.join("\n"),
    name: SENDER_NAME,
    replyTo: NOTIFY_EMAIL
  });
  return true;
}

function notifyWes(email, firstName, source, page, result) {
  var subject = result.isNew
    ? "New Squish " + (result.status === "Subscribed" ? "subscriber" : "printables request") + ": " + (firstName || email)
    : "Squish sign-up (already on sheet, now " + result.status + "): " + email;
  MailApp.sendEmail({
    to: NOTIFY_EMAIL,
    subject: subject,
    body:
      "Name: " + (firstName || "(not given)") + "\n" +
      "Email: " + email + "\n" +
      "Status: " + result.status + "\n" +
      "Source: " + source + "\n" +
      "Page: " + page + "\n" +
      "Time: " + new Date() + "\n\n" +
      "Sheet: " + SpreadsheetApp.getActiveSpreadsheet().getUrl()
  });
}

/** Unsubscribe link for an address; use it in every monthly email you send. */
function unsubscribeUrl(email) {
  return EXEC_URL + "?u=" + encodeURIComponent(email) + "&t=" + tokenFor(email);
}

function tokenFor(email) {
  var props = PropertiesService.getScriptProperties();
  var secret = props.getProperty("UNSUB_SECRET");
  if (!secret) {
    secret = Utilities.getUuid();
    props.setProperty("UNSUB_SECRET", secret);
  }
  var bytes = Utilities.computeHmacSha256Signature(email, secret);
  return Utilities.base64EncodeWebSafe(bytes).slice(0, 22);
}

function getSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  return SHEET_NAME ? ss.getSheetByName(SHEET_NAME) : ss.getSheets()[0];
}

function ensureHeaders(sheet) {
  var width = HEADERS.length;
  var first = sheet.getRange(1, 1, 1, width).getValues()[0];
  for (var i = 0; i < width; i++) {
    if (!first[i]) {
      sheet.getRange(1, i + 1).setValue(HEADERS[i]);
    }
  }
}

function findRow(sheet, email) {
  var last = sheet.getLastRow();
  if (last < 2) {
    return 0;
  }
  var emails = sheet.getRange(2, COL.email, last - 1, 1).getValues();
  for (var i = 0; i < emails.length; i++) {
    if (String(emails[i][0]).trim().toLowerCase() === email) {
      return i + 2;
    }
  }
  return 0;
}

/** Run once from the editor to approve permissions and add the new columns. */
function testSetup() {
  ensureHeaders(getSheet());
  console.log("Quota left today: " + MailApp.getRemainingDailyQuota());
  console.log("Example unsubscribe link: " + unsubscribeUrl(NOTIFY_EMAIL));
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function htmlPage(message) {
  return HtmlService.createHtmlOutput(
    '<div style="font-family:sans-serif;max-width:420px;margin:60px auto;text-align:center;color:#39205C">' +
    "<h2>Learning with Squish</h2><p>" + message + "</p>" +
    '<p><a href="' + SITE + '">Back to the books</a></p></div>'
  ).setTitle("Squish List");
}
