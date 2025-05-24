document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const symptomsInput = document.getElementById('symptomsInput');
    const resultsSection = document.getElementById('resultsSection');
    const emergencyWarning = document.getElementById('emergencyWarning');
    const severityProgress = document.getElementById('severityProgress');
    const severityText = document.getElementById('severityText');
    const severityDescription = document.getElementById('severityDescription');
    const severityBadge = document.getElementById('severityBadge');
    const specialistName = document.getElementById('specialistName');
    const specialistDescription = document.getElementById('specialistDescription');
    const specialistIcon = document.getElementById('specialistIcon');
    const urgencyTextEl = document.getElementById('urgencyText');
    const urgencyDescriptionEl = document.getElementById('urgencyDescription');
    const urgencyIconEl = document.getElementById('urgencyIcon');
    const aiResponseEl = document.getElementById('aiResponse');
    const conditionsList = document.getElementById('conditionsList');
    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');
    const originalComplaintTextEl = document.getElementById('originalComplaintText');
    const processedComplaintTextEl = document.getElementById('processedComplaintText');

    const backendUrl = 'http://127.0.0.1:5000/predict';

    const specialistsData = {
        'терапевт': { icon: 'fas fa-stethoscope', description: 'Кеңири профилдеги ооруларды аныктоо жана дарылоо боюнча жалпы адис.' },
        'невролог': { icon: 'fas fa-brain', description: 'Нерв системасынын ооруларын, баш ооруну, уйкусуздукту дарылайт.' },
        'кардиолог': { icon: 'fas fa-heartbeat', description: 'Жүрөк жана кан тамыр оорулары боюнча адис.' },
        'гастроэнтеролог': { icon: 'fas fa-stomach', description: 'Ашказан-ичеги жолдорунун ооруларына адистешкен дарыгер.' },
        'пульмонолог': { icon: 'fas fa-lungs', description: 'Өпкө жана дем алуу жолдорунун оорулары боюнча адис.' },
        'уролог': { icon: 'fas fa-bladder', description: 'Заара чыгаруу системасынын ооруларын дарылайт.' },
        'эндокринолог': { icon: 'fas fa-seedling', description: 'Гормоналдык бузулууларды жана зат алмашуу ооруларын дарылайт.' },
        'дерматолог': { icon: 'fas fa-allergies', description: 'Тери, чач жана тырмак оорулары боюнча адис.' },
        'офтальмолог': { icon: 'fas fa-eye', description: 'Көз ооруларына адистешкен дарыгер.' },
        'лор': { icon: 'fas fa-ear-listen', description: 'Кулак, мурун, тамак оорулары боюнча адис (отоларинголог).' },
        'хирург': { icon: 'fas fa-scalpel', description: 'Операциялык кийлигишүүнү талап кылган ооруларды дарылоочу адис.' },
        'травматолог': { icon: 'fas fa-bone', description: 'Жаракаттарды, сыныктарды жана таяныч-кыймыл аппаратынын ооруларын дарылайт.' },
        'педиатр': { icon: 'fas fa-baby', description: 'Балдардын ден соолугу жана оорулары боюнча адис.' },
        'гинеколог': { icon: 'fas fa-venus', description: 'Аялдардын ден соолугу жана репродуктивдик системасы боюнча адис.' },
        'стоматолог': { icon: 'fas fa-tooth', description: 'Тиш жана ооз көңдөйүнүн ооруларын дарылоочу дарыгер.' },
        'не определено': { icon: 'fas fa-question-circle', description: 'Дарыгер так аныкталган жок. Сураныч, белгилериңизди кененирээк жазыңыз.' }
    };

    const urgencyMapping = {
        'Красный': { key: 'immediate', severityScore: 95, badgeColor: 'bg-red-600 text-red-100', text: 'Критикалык' },
        'Оранжевый': { key: 'urgent', severityScore: 75, badgeColor: 'bg-orange-600 text-orange-100', text: 'Жогорку' },
        'Желтый': { key: 'soon', severityScore: 50, badgeColor: 'bg-yellow-600 text-yellow-100', text: 'Орточо' },
        'Зеленый': { key: 'routine', severityScore: 25, badgeColor: 'bg-green-600 text-green-100', text: 'Төмөн' },
        'Не определено': { key: 'unknown', severityScore: 0, badgeColor: 'bg-gray-600 text-gray-100', text: 'Белгисиз' }
    };

    const urgencyLevelsDetails = {
        'immediate': {
            text: 'Дароо жардам',
            description: 'Өмүргө коркунуч туулушу мүмкүн. Тез арада медициналык жардам талап кылынат!',
            icon: 'fas fa-ambulance text-red-500'
        },
        'urgent': {
            text: 'Шашылыш',
            description: '24 сааттын ичинде дарыгерге кайрылуу сунушталат.',
            icon: 'fas fa-exclamation-triangle text-orange-500'
        },
        'soon': {
            text: 'Жакын арада',
            description: 'Жакынкы күндөрү дарыгерге кайрылуу пландаштырылышы керек.',
            icon: 'fas fa-clock text-yellow-500'
        },
        'routine': {
            text: 'Пландуу',
            description: 'Пландуу түрдө дарыгерге кайрылсаңыз болот.',
            icon: 'fas fa-calendar-check text-green-500'
        },
        'unknown': {
            text: 'Белгисиз',
            description: 'Шашылыш деңгээли аныкталган жок. Белгилериңизди толугураак жазыңыз.',
            icon: 'fas fa-question-circle text-gray-500'
        }
    };

    const sampleConditions = [
        { name: "Мигрень (баш катуу оорусу)", probability: 65 },
        { name: "Гипертониялык криз (кан басымдын кескин көтөрүлүшү)", probability: 30 },
        { name: "Вирустук инфекция", probability: 45 },
        { name: "Өнөкөт чарчоо синдрому", probability: 25 }
    ];

    analyzeBtn.addEventListener('click', async function() {
        const symptoms = symptomsInput.value.trim();

        if (!symptoms) {
            alert('Сураныч, белгилериңизди жазыңыз.');
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Талдап жатат...';
        resultsSection.classList.add('hidden');
        emergencyWarning.classList.add('hidden');
        errorSection.classList.add('hidden');

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ complaint: symptoms }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `Сервер катасы: ${response.status}` }));
                throw new Error(errorData.error || `HTTP катасы: ${response.status}`);
            }

            const data = await response.json();
            resultsSection.classList.remove('hidden');


            const backendUrgency = data.predicted_urgency || 'Не определено';
            const urgencyInfo = urgencyMapping[backendUrgency] || urgencyMapping['Не определено'];

            severityProgress.style.width = urgencyInfo.severityScore + '%';
            severityText.textContent = urgencyInfo.text;
            severityDescription.textContent = urgencyLevelsDetails[urgencyInfo.key].description;
            severityBadge.className = `px-4 py-1 rounded-full text-sm font-medium ${urgencyInfo.badgeColor}`;
            severityBadge.classList.remove('hidden');

            if (urgencyInfo.key === 'immediate') {
                emergencyWarning.classList.remove('hidden');
            } else {
                emergencyWarning.classList.add('hidden');
            }

            const urgencyData = urgencyLevelsDetails[urgencyInfo.key];
            urgencyTextEl.textContent = urgencyData.text;
            urgencyDescriptionEl.textContent = urgencyData.description;
            urgencyIconEl.className = urgencyData.icon;

            const backendSpecialistRaw = data.predicted_specialist || 'не определено';
            const backendSpecialist = backendSpecialistRaw.toLowerCase();
            const specialistInfo = specialistsData[backendSpecialist] || specialistsData['не определено'];

            let specialistDisplayName = backendSpecialistRaw.charAt(0).toUpperCase() + backendSpecialistRaw.slice(1);


            specialistName.textContent = specialistDisplayName;
            specialistDescription.textContent = specialistInfo.description;
            specialistIcon.className = specialistInfo.icon;

            originalComplaintTextEl.textContent = `Баштапкы арыз: ${data.complaint_original || symptoms}`;
            processedComplaintTextEl.textContent = `Иштелип чыккан арыз: ${data.complaint_processed || 'маалымат жок'}`;

            let aiMessage = `Талдоо аяктады. Сунушталган шашылыш деңгээли: ${urgencyData.text}. `;
            aiMessage += `Болжолдонгон дарыгер: ${specialistDisplayName}. `;

            if (data.urgency_confidence) {
                const topUrgencyConf = data.urgency_confidence[data.predicted_urgency] || 0;
                aiMessage += `Шашылыш деңгээлин баалоодогу ишенимдүүлүк: ${(topUrgencyConf * 100).toFixed(0)}%. `;
            }
            if (data.specialist_confidence) {
                const topSpecialistConf = data.specialist_confidence[data.predicted_specialist] || 0;
                aiMessage += `Дарыгерди тандоодогу ишенимдүүлүк: ${(topSpecialistConf * 100).toFixed(0)}%. `;
            }
            if (data.error_message) {
                aiMessage += `Системанын эскертүүсү: ${data.error_message}. `;
            }

            typeWriter(aiMessage, aiResponseEl);

            conditionsList.innerHTML = '';
            sampleConditions.forEach(condition => {
                const conditionElement = document.createElement('div');
                conditionElement.className = 'bg-gray-800 rounded-lg p-3 flex items-center';
                const probabilityBar = document.createElement('div');
                probabilityBar.className = 'w-12 h-2 bg-gray-700 rounded-full mr-3 overflow-hidden';
                const probabilityFill = document.createElement('div');
                probabilityFill.className = 'h-full bg-cyan-500';
                probabilityFill.style.width = condition.probability + '%';
                probabilityBar.appendChild(probabilityFill);
                conditionElement.innerHTML = `
                    <div class="flex-shrink-0 w-8 h-8 rounded-full bg-cyan-900 flex items-center justify-center mr-3">
                        <i class="fas fa-disease text-cyan-400 text-sm"></i>
                    </div>
                    <div class="flex-grow">
                        <h4 class="font-medium">${condition.name}</h4>
                        <div class="flex items-center mt-1">
                            ${probabilityBar.outerHTML}
                            <span class="text-xs text-gray-400 ml-2">${condition.probability}%</span>
                        </div>
                    </div>`;
                conditionsList.appendChild(conditionElement);
            });

        } catch (error) {
            console.error('Талдоо учурунда ката кетти:', error);
            errorMessageEl.textContent = `Ката: ${error.message}. Сураныч, кайталап көрүңүз же сервер иштеп жатканын текшериңиз.`;
            errorSection.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-redo mr-2"></i> Кайра талдоо';
        }
    });

    function typeWriter(text, element, speed = 20, i = 0) {
        if (i < text.length) {
            element.textContent = text.substring(0, i + 1);
            element.classList.add('typing-animation');
            setTimeout(function() {
                typeWriter(text, element, speed, i + 1);
            }, speed);
        } else {
            element.classList.remove('typing-animation');
        }
    }
});