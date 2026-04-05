/**
 * squashAttest.groovy — Jenkins Shared Library step for Squash attestation.
 *
 * Runs the full Squash attestation pipeline inside a Jenkins stage, publishes
 * the generated artifacts as build artifacts, and optionally fails the build
 * on compliance violations.
 *
 * Usage in Jenkinsfile:
 *
 *   @Library('jenkins-pipeline') _
 *
 *   stage('Attest Model') {
 *     squashAttest(
 *       modelPath: './output/llama-3.1-8b',
 *       policies:  ['eu-ai-act', 'nist-ai-rmf'],
 *       sign:      true,
 *     )
 *   }
 *
 * Parameters:
 *   modelPath         (required) Path to model directory or file.
 *   outputDir         Output directory for artifacts; defaults to modelPath dir.
 *   policies          List of policy names (default: ['enterprise-strict']).
 *   sign              Sign via Sigstore (default: false).
 *   failOnViolation   Fail the build on any error-severity violation (default: true).
 *   skipScan          Skip security scanner (default: false).
 *   squishVersion     Squish version to install (default: '' = latest).
 *   archiveArtifacts  Archive attestation artifacts as build artifacts (default: true).
 *   pythonExecutable  Python binary to use (default: 'python3').
 */
def call(Map params = [:]) {
    def modelPath        = params.get('modelPath')        ?: error('squashAttest: modelPath is required')
    def outputDir        = params.get('outputDir')        ?: ''
    def policies         = params.get('policies')         ?: ['enterprise-strict']
    def sign             = params.get('sign')             ?: false
    def failOnViolation  = params.get('failOnViolation')  != null ? params.failOnViolation : true
    def skipScan         = params.get('skipScan')         ?: false
    def squishVersion    = params.get('squishVersion')    ?: ''
    def doArchive        = params.get('archiveArtifacts') != null ? params.archiveArtifacts : true
    def python           = params.get('pythonExecutable') ?: 'python3'

    stage('Squash: AI-SBOM Attestation') {
        // Install squish with squash optional dependencies
        def installCmd = squishVersion
            ? "${python} -m pip install --quiet 'squish[squash]==${squishVersion}'"
            : "${python} -m pip install --quiet 'squish[squash]'"

        sh installCmd

        // Build CLI command
        def cli = "squash attest --model-path '${modelPath}'"
        policies.each { p -> cli += " --policy '${p}'" }
        if (sign)            cli += ' --sign'
        if (skipScan)        cli += ' --skip-scan'
        if (outputDir)       cli += " --output-dir '${outputDir}'"
        cli += ' --json-result /tmp/squash-result.json'

        // Run attestation; capture result without immediately failing
        def rc = sh(script: cli, returnStatus: true)

        // Parse result JSON for structured reporting
        def resultJson = readJSON file: '/tmp/squash-result.json'
        def passed     = resultJson.get('passed', false)
        def scanStatus = resultJson.get('scan_status', 'skipped')
        def artifacts  = resultJson.get('artifacts', [:])

        echo "Squash: passed=${passed}  scan=${scanStatus}"
        resultJson.get('policy_results', [:]).each { policy, pr ->
            echo "  [${pr.passed ? 'PASS' : 'FAIL'}] ${policy}: ${pr.error_count} error(s), ${pr.warning_count} warning(s)"
        }

        // Archive attestation artifacts
        if (doArchive) {
            def toArchive = []
            if (artifacts.cyclonedx)     toArchive << artifacts.cyclonedx
            if (artifacts.spdx_json)     toArchive << artifacts.spdx_json
            if (artifacts.master_record) toArchive << artifacts.master_record
            if (toArchive) {
                archiveArtifacts artifacts: toArchive.join(', '), allowEmptyArchive: true
            }
        }

        // Fail the build if requested and attestation failed
        if (failOnViolation && !passed) {
            error("Squash attestation FAILED for '${modelPath}'. " +
                  "Scan: ${scanStatus}. See archived squash-attest.json for details.")
        }
    }
}
