(() => {
  const formatVector = (values) => `[${values.map((value) => value.toFixed(1)).join(', ')}]`;

  document.querySelectorAll('[data-step-through]').forEach((walkthrough) => {
    const dataElement = walkthrough.querySelector('[data-step-data]');
    if (!dataElement) return;

    const data = JSON.parse(dataElement.textContent);
    const stages = data.stages || [];
    if (stages.length === 0) return;

    let stepIndex = 0;
    const tokens = Array.from(walkthrough.querySelectorAll('[data-attention-token]'));
    const keys = Array.from(walkthrough.querySelectorAll('[data-attention-key]'));
    const weights = Array.from(walkthrough.querySelectorAll('[data-attention-weight]'));
    const query = walkthrough.querySelector('[data-attention-query]');
    const output = walkthrough.querySelector('[data-attention-output]');
    const counter = walkthrough.querySelector('[data-step-counter]');
    const title = walkthrough.querySelector('[data-step-title]');
    const description = walkthrough.querySelector('[data-step-description]');
    const previous = walkthrough.querySelector('[data-step-action="previous"]');
    const next = walkthrough.querySelector('[data-step-action="next"]');

    const render = () => {
      const stage = stages[stepIndex];
      counter.textContent = `Step ${stepIndex + 1} of ${stages.length}`;
      title.textContent = stage.title;
      description.textContent = stage.explanation;
      query.textContent = formatVector(stage.query);
      output.textContent = stage.output;

      tokens.forEach((token, index) => token.classList.toggle('is-active', index === stage.active_token));
      keys.forEach((key, index) => {
        key.textContent = `Key ${formatVector(stage.keys[index])}`;
        key.classList.toggle('is-strongest', stage.scores[index] === Math.max(...stage.scores));
      });
      weights.forEach((weight, index) => {
        const percentage = Math.round(stage.weights[index] * 100);
        weight.style.setProperty('--attention-weight', `${percentage}%`);
        weight.textContent = `${data.tokens[index]} ${percentage}%`;
      });

      previous.disabled = stepIndex === 0;
      next.disabled = stepIndex === stages.length - 1;
    };

    walkthrough.addEventListener('click', (event) => {
      const action = event.target.closest('[data-step-action]')?.dataset.stepAction;
      if (action === 'previous' && stepIndex > 0) stepIndex -= 1;
      if (action === 'next' && stepIndex < stages.length - 1) stepIndex += 1;
      if (action === 'reset') stepIndex = 0;
      render();
    });

    render();
  });
})();
