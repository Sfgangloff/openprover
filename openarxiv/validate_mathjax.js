/**
 * Validate MathJax rendering of markdown fields.
 *
 * Reads JSON from stdin: [{field: "statement", text: "..."}, ...]
 * Outputs JSON to stdout: {errors: [{field: "statement", expression: "...", message: "..."}]}
 *
 * Uses mathjax-full to render each field as an HTML document (same config as
 * the web frontend) and checks for mjx-merror nodes in the output.
 */

const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');

// Use all packages except noundefined/noerrors so that parse errors surface
// as merror nodes instead of being silently swallowed.
const packages = AllPackages.filter(p => p !== 'noundefined' && p !== 'noerrors');

const adaptor = liteAdaptor();
const handler = RegisterHTMLHandler(adaptor);

function validate(fields) {
  const errors = [];

  for (const {field, text} of fields) {
    // Wrap in a minimal HTML document so MathJax's FindTeX can scan it.
    const htmlContent = `<html><body>${text}</body></html>`;

    const tex = new TeX({
      inlineMath: [['$', '$']],
      displayMath: [['$$', '$$']],
      processEscapes: true,
      packages,
    });
    const svg = new SVG({fontCache: 'none'});
    const doc = mathjax.document(htmlContent, {
      InputJax: tex,
      OutputJax: svg,
    });
    doc.render();

    for (const item of doc.math) {
      item.root.walkTree((node) => {
        if (node.isKind('merror')) {
          const msg = node.attributes.get('data-mjx-error') || 'unknown error';
          errors.push({field, expression: item.math, message: msg});
        }
      });
    }

    // Clean up the document so it doesn't leak into the next iteration
    mathjax.handlers.unregister(handler);
    RegisterHTMLHandler(adaptor);
  }

  return errors;
}

// Read all of stdin, parse JSON, validate, output JSON.
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  try {
    const fields = JSON.parse(input);
    const errors = validate(fields);
    process.stdout.write(JSON.stringify({errors}) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({errors: [{field: '_parse', expression: '', message: e.message}]}) + '\n');
    process.exit(1);
  }
});
