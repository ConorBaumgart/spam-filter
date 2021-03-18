<template>
    <v-container id="message-answer" >
        <h1 class="mb-4">Spam or Ham Report</h1>
        <v-card v-for="spam in spamData" :key="spam.type" class="pa-5" raised>
            <h1>{{ spam.spam }}</h1>
            <h3>Chance it is not spam: {{ spam.chance_no }}</h3>
            <h3>Chance it is spam: {{ spam.chance_yes }}</h3>
        </v-card>
    </v-container>
</template>

<script>

export default {
    name: "SpamAnswer",
    props: [
        "spamData"
    ]
}
</script>

<style>
#message-answer {
  max-width: 600px;
}
</style>